"""
memory_bank.py — Surprise-Driven Memory Bank

存储 VAE latent 帧 + camera pose embedding，按 Surprise score 筛选，
用 cosine similarity 检索最相关的历史帧供 MemoryCrossAttention 使用。

依赖：PyTorch（无 lingbot-world 依赖，独立可测试）
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryFrame:
    """单帧记忆条目。

    Attributes:
        pose_emb: 当前帧的 camera pose 在模型空间的嵌入，[dim]，
                  由 WanModel 的 c2ws_plucker_emb 经过 mean-pool 得到。
                  用作 MemoryCrossAttention 的 Key/Value。
        latent:   VAE encoded latent，[z_dim, h, w]，
                  用于 NFPHead 的预测目标。
        surprise_score: NFPHead 计算的 cosine distance（越大越"意外"）。
        timestep: 原始视频帧索引，用于 temporal ordering。
    """
    pose_emb: Tensor          # [dim]
    latent: Tensor            # [z_dim, h, w]
    surprise_score: float
    timestep: int


class MemoryBank:
    """Surprise-Driven Memory Bank。

    容量为 max_size，存满后替换 surprise_score 最低的帧。
    检索时用 query 的 pose_emb 与所有存储帧的 pose_emb 做 cosine similarity，
    返回 top-k 帧的 pose_emb 作为 MemoryCrossAttention 的 Key/Value 输入。

    使用方式（推理循环）：
        bank = MemoryBank(max_size=8)
        # 每帧生成后：
        bank.update(pose_emb, latent, surprise_score, timestep)
        # 下一帧生成前：
        memory_states = bank.retrieve(query_pose_emb, top_k=4)  # [top_k, dim]
    """

    def __init__(self, max_size: int = 8):
        """
        Args:
            max_size: Memory Bank 的最大容量 K，默认 8（WorldMem reference_length=8）
        """
        self.max_size = max_size
        self.frames: List[MemoryFrame] = []

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        pose_emb: Tensor,
        latent: Tensor,
        surprise_score: float,
        timestep: int,
    ) -> None:
        """存入一帧。若已满，替换 surprise_score 最低的帧。

        Args:
            pose_emb:       [dim] 当前帧的 pose embedding（已 mean-pool 到 1D）
            latent:         [z_dim, h, w] VAE latent
            surprise_score: NFPHead 计算的 per-frame surprise（0~2 之间，越大越值得存）
            timestep:       当前帧在原始视频中的帧索引
        """
        new_frame = MemoryFrame(
            pose_emb=pose_emb.detach().cpu(),
            latent=latent.detach().cpu(),
            surprise_score=float(surprise_score),
            timestep=int(timestep),
        )

        if len(self.frames) < self.max_size:
            self.frames.append(new_frame)
            logger.debug(
                "MemoryBank: added frame t=%d, surprise=%.4f, size=%d/%d",
                timestep, surprise_score, len(self.frames), self.max_size,
            )
        else:
            # 替换 surprise_score 最低的帧
            min_idx = min(range(len(self.frames)),
                          key=lambda i: self.frames[i].surprise_score)
            evicted = self.frames[min_idx]
            if surprise_score > evicted.surprise_score:
                self.frames[min_idx] = new_frame
                logger.debug(
                    "MemoryBank: replaced t=%d(s=%.4f) with t=%d(s=%.4f)",
                    evicted.timestep, evicted.surprise_score,
                    timestep, surprise_score,
                )
            else:
                logger.debug(
                    "MemoryBank: frame t=%d(s=%.4f) not stored (min stored=%.4f)",
                    timestep, surprise_score, evicted.surprise_score,
                )

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_pose_emb: Tensor,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Tensor]:
        """按 pose cosine similarity 检索最相关的 top-k 帧。

        Args:
            query_pose_emb: [dim] 当前帧的 pose embedding，用于检索
            top_k:          返回帧数，None 表示返回全部
            device:         输出 tensor 放置的设备，None 时跟随 query

        Returns:
            memory_states: [k, dim] 检索到的帧的 pose_emb，
                           k = min(top_k, len(self.frames))
                           若 bank 为空返回 None
        """
        if not self.frames:
            return None

        device = device or query_pose_emb.device
        k = min(top_k, len(self.frames)) if top_k is not None else len(self.frames)

        all_pose_embs = torch.stack(
            [f.pose_emb for f in self.frames]
        ).to(device)  # [K, dim]

        # cosine similarity: [K]
        sims = F.cosine_similarity(
            query_pose_emb.unsqueeze(0).to(device),  # [1, dim]
            all_pose_embs,                            # [K, dim]
            dim=-1,
        )

        _, indices = torch.topk(sims, k=k)
        memory_states = torch.stack(
            [self.frames[i].pose_emb for i in indices.tolist()]
        ).to(device)  # [k, dim]

        return memory_states

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_states(
        self, device: Optional[torch.device] = None
    ) -> Optional[Tensor]:
        """返回所有存储帧的 pose_emb，[K, dim]。

        适用于不需要检索筛选、直接把全部 memory 传入 cross-attention 的场景。
        若 bank 为空返回 None。
        """
        if not self.frames:
            return None
        device = device or self.frames[0].pose_emb.device
        return torch.stack([f.pose_emb for f in self.frames]).to(device)

    def size(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        """清空 Memory Bank（换 episode 时调用）。"""
        self.frames.clear()
        logger.info("MemoryBank: cleared.")

    def __repr__(self) -> str:
        scores = [f"{f.surprise_score:.3f}" for f in self.frames]
        return (
            f"MemoryBank(size={len(self.frames)}/{self.max_size}, "
            f"surprise=[{', '.join(scores)}])"
        )
