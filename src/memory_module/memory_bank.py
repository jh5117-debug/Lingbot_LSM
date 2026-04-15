"""
memory_bank.py — Memory Bank 模块

当前实现（v2，已完成）：单层 MemoryBank
  - MemoryFrame dataclass：单帧记忆条目，含 pose_emb / visual_emb / latent / surprise_score / timestep / chunk_id / age / （待新增：semantic_key）
  - MemoryBank：固定容量 K，surprise-driven 写入（evict 最低 surprise 帧）
  - retrieve()：query_pose_emb 与存储帧做 cosine similarity，返回 (pose_embs [k,5120], visual_embs [k,5120])

下一步实现（待实现，参见 state/design_gap.md）：三层 ThreeTierMemoryBank
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ShortTermBank   │ FIFO，容量 2，强制存最近帧，保证 chunk 连续性          │
  │  MediumTermBank  │ 容量 8，高 surprise 帧，age decay eviction           │
  │  LongTermBank    │ 容量 8-16，stable（低 surprise）且 novel（语义新颖）帧 │
  └─────────────────────────────────────────────────────────────────────┘

  新增：semantic_key = cross_attn.norm_k(cross_attn.k(pose_emb)).detach()
        LongTermBank novelty check：max cosine_sim(semantic_key, existing_keys) < NOVELTY_THRESHOLD
        LongTermBank eviction：移除语义最冗余帧（max sim 最大者）

  混合检索预算（Hybrid Retrieval Budget）：Short 2 + Medium top-3 + Long top-2 = 7 帧
  retrieve() 返回：(key_states [7,5120], value_states [7,5120])

依赖：PyTorch（无 lingbot-world 依赖，独立可测试）
设计参考：
  - WorldMem：Memory Bank 结构 + FOV pose 检索
  - Cambrian-S：NFPHead Surprise 机制（见 nfp_head.py）
  - HyDRA（Out of Sight but Not Out of Mind）：semantic_key 借鉴 K 投影特征提取器思路
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
        pose_emb:   当前帧的 camera pose 在模型空间的嵌入，[dim]，
                    由 WanModel 的 c2ws_plucker_emb 经过 mean-pool 得到。
                    用作 MemoryCrossAttention 的 Key（FOV 路由）。
        latent:     VAE encoded latent，[z_dim, h, w]，
                    用于 NFPHead 的预测目标。
        surprise_score: NFPHead 计算的 cosine distance（越大越"意外"）。
        timestep:   原始视频帧索引，用于 temporal ordering。
        visual_emb: [dim=5120]，VAE latent 投影到模型空间的视觉嵌入；
                    用作 MemoryCrossAttention 的 Value（视觉内容）。
                    若 None 则 retrieve() 退化为 pose_emb 做 V（向后兼容）。
                    # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
    """
    pose_emb: Tensor                      # [dim]
    latent: Tensor                         # [z_dim, h, w]
    surprise_score: float
    timestep: int
    visual_emb: Optional[Tensor] = None   # [dim=5120]，MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
    chunk_id: int = 0    # 所属 chunk 编号，授权新增（Orchestrator 2026-04-02）
    age: int = 0         # 自写入以来经历的 chunk 数，授权新增（Orchestrator 2026-04-02）


class MemoryBank:
    """Surprise-Driven Memory Bank。

    容量为 max_size，存满后替换 surprise_score 最低的帧。
    检索时用 query 的 pose_emb 与所有存储帧的 pose_emb 做 cosine similarity，
    返回 top-k 帧的 pose_emb 作为 MemoryCrossAttention 的 Key/Value 输入。

    使用方式（推理循环）：
        bank = MemoryBank(max_size=8)
        # 每帧生成后：
        bank.update(pose_emb, latent, surprise_score, timestep, visual_emb=visual_emb)
        # 下一帧生成前（MODIFIED: F-03/F5 fix — 返回 tuple）：
        retrieved = bank.retrieve(query_pose_emb, top_k=4)  # (pose_embs [k,dim], visual_embs [k,dim])
        if retrieved is not None:
            key_states, value_states = retrieved
    """

    def __init__(self, max_size: int = 8):
        """
        Args:
            max_size: Memory Bank 的最大容量 K，默认 8（WorldMem reference_length=8）
        """
        self.max_size = max_size
        self.frames: List[MemoryFrame] = []
        # 操作统计（用于 W&B 日志）
        self.store_count: int = 0
        self.reject_count: int = 0
        self.evict_count: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        pose_emb: Tensor,
        latent: Tensor,
        surprise_score: float,
        timestep: int,
        visual_emb: Optional[Tensor] = None,  # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        chunk_id: int = 0,   # 新增，默认 0（向后兼容）
    ) -> None:
        """存入一帧。若已满，替换 surprise_score 最低的帧。

        Args:
            pose_emb:       [dim] 当前帧的 pose embedding（已 mean-pool 到 1D）
            latent:         [z_dim, h, w] VAE latent
            surprise_score: NFPHead 计算的 per-frame surprise（0~2 之间，越大越值得存）
            timestep:       当前帧在原始视频中的帧索引
            visual_emb:     [dim=5120] VAE latent 投影到模型空间的视觉嵌入（可选）；
                            None 时 retrieve() 退化为 pose_emb 做 V（向后兼容）
                            # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
            chunk_id:       所属 chunk 编号（Feature 3 新增，默认 0）
        """
        # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        new_frame = MemoryFrame(
            pose_emb=pose_emb.detach().cpu(),
            latent=latent.detach().cpu(),
            surprise_score=float(surprise_score),
            timestep=int(timestep),
            visual_emb=visual_emb.detach().cpu() if visual_emb is not None else None,
            chunk_id=int(chunk_id),
            age=0,
        )

        if len(self.frames) < self.max_size:
            self.frames.append(new_frame)
            self.store_count += 1
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
                self.evict_count += 1
                self.store_count += 1
                logger.debug(
                    "MemoryBank: replaced t=%d(s=%.4f) with t=%d(s=%.4f)",
                    evicted.timestep, evicted.surprise_score,
                    timestep, surprise_score,
                )
            else:
                self.reject_count += 1
                logger.debug(
                    "MemoryBank: frame t=%d(s=%.4f) not stored (min stored=%.4f)",
                    timestep, surprise_score, evicted.surprise_score,
                )

    def increment_age(self) -> None:
        """在每个新 chunk 生成前调用，所有已存储帧 age +1。"""
        for frame in self.frames:
            frame.age += 1

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_pose_emb: Tensor,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """按 pose cosine similarity 检索最相关的 top-k 帧。

        MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        返回类型从 Optional[Tensor] 改为 Optional[Tuple[Tensor, Tensor]]：
          - (pose_embs [k, dim], visual_embs [k, dim])
          - pose_embs 用作 cross-attention 的 K（FOV 路由）
          - visual_embs 用作 cross-attention 的 V（视觉内容）
          - 若所有帧 visual_emb is None，则 visual_embs = pose_embs（退化，向后兼容）

        Args:
            query_pose_emb: [dim] 当前帧的 pose embedding，用于检索
            top_k:          返回帧数，None 表示返回全部
            device:         输出 tensor 放置的设备，None 时跟随 query

        Returns:
            (pose_embs, visual_embs): 各 [k, dim]，k = min(top_k, len(self.frames))
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
        idx_list = indices.tolist()

        # MODIFIED: F-03/F5 fix — K = pose_embs, V = visual_embs
        pose_embs = torch.stack(
            [self.frames[i].pose_emb for i in idx_list]
        ).to(device)   # [k, dim]

        # 若任何帧有 visual_emb，则为每帧填充（None 退化为 pose_emb）
        if any(self.frames[i].visual_emb is not None for i in idx_list):
            visual_embs = torch.stack([
                self.frames[i].visual_emb if self.frames[i].visual_emb is not None
                else self.frames[i].pose_emb
                for i in idx_list
            ]).to(device)   # [k, dim]
        else:
            visual_embs = pose_embs   # 退化路径（向后兼容）

        return pose_embs, visual_embs

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
        self.store_count = 0
        self.reject_count = 0
        self.evict_count = 0
        logger.info("MemoryBank: cleared.")

    def get_stats(self) -> dict:
        """返回 W&B 可记录的统计字典。"""
        surprises = [f.surprise_score for f in self.frames]
        ages = [f.age for f in self.frames]
        return {
            "memory/bank_size": float(len(self.frames)),
            "memory/store_count": float(self.store_count),
            "memory/reject_count": float(self.reject_count),
            "memory/evict_count": float(self.evict_count),
            "memory/surprise_mean": float(sum(surprises) / max(len(surprises), 1)),
            "memory/surprise_max": float(max(surprises)) if surprises else 0.0,
            "memory/surprise_min": float(min(surprises)) if surprises else 0.0,
            "memory/age_mean": float(sum(ages) / max(len(ages), 1)),
        }

    def __repr__(self) -> str:
        scores = [f"{f.surprise_score:.3f}" for f in self.frames]
        return (
            f"MemoryBank(size={len(self.frames)}/{self.max_size}, "
            f"surprise=[{', '.join(scores)}])"
        )
