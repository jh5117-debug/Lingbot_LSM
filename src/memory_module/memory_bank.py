"""
memory_bank.py — Surprise-Driven Memory Bank (v2)

存储模型空间 pose embedding (key_state) + 视觉嵌入 (value_visual)，
按 Surprise score 筛选写入，用 cosine similarity 检索最相关的历史帧
供 MemoryCrossAttention 使用。

v2 变更（对应 MEMORY_FIX_REVIEW_AND_WANDB_PLAN.md 任务 C+D）：
  - MemoryFrame 增加 value_visual / chunk_id / age 字段
  - retrieve() 返回结构化 dict（含 similarities）
  - 增加操作统计：store_count / reject_count / evict_count
  - 新增 get_stats() 用于 W&B 日志

依赖：PyTorch（无 lingbot-world 依赖，独立可测试）
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    """单帧记忆条目（v2: 分离 key_state / value_visual）。

    Attributes:
        key_state:      检索用 key，来自 get_projected_frame_embs()，[dim]
        value_visual:   注入 cross-attention 的 value，[dim]
                        当前阶段 key_state == value_visual（WorldMem fallback 模式 L297-298）
        surprise_score: NFPHead 计算的 per-frame surprise（越大越值得存）
        timestep:       原始视频帧索引
        chunk_id:       所属 chunk 的编号
        age:            自写入以来经历的 chunk 数（每次新 chunk 时 +1）
    """
    key_state: Tensor       # [dim]
    value_visual: Tensor    # [dim]
    surprise_score: float
    timestep: int
    chunk_id: int = 0
    age: int = 0


class MemoryBank:
    """Surprise-Driven Memory Bank (v2)。

    容量为 max_size，存满后替换 surprise_score 最低的帧。
    检索时用 query 的 key_state 与所有存储帧的 key_state 做 cosine similarity，
    返回 top-k 帧的结构化结果。

    使用方式（推理循环）：
        bank = MemoryBank(max_size=8)
        # 每 chunk 生成后：
        bank.update(key_state, value_visual, surprise_score, timestep, chunk_id)
        # 下一 chunk 生成前：
        result = bank.retrieve(query_key_state, top_k=4)
        # result["value_visuals"]  → [k, dim]  传入 MemoryCrossAttention
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
        key_state: Tensor,
        value_visual: Tensor,
        surprise_score: float,
        timestep: int,
        chunk_id: int = 0,
    ) -> None:
        """存入一帧。若已满，替换 surprise_score 最低的帧。

        Args:
            key_state:      [dim] 检索用嵌入（来自 get_projected_frame_embs）
            value_visual:   [dim] 视觉内容嵌入（当前阶段 = key_state）
            surprise_score: per-frame surprise（0~2 之间，越大越值得存）
            timestep:       当前帧在原始视频中的帧索引
            chunk_id:       当前 chunk 编号
        """
        new_frame = MemoryFrame(
            key_state=key_state.detach().cpu(),
            value_visual=value_visual.detach().cpu(),
            surprise_score=float(surprise_score),
            timestep=int(timestep),
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
        """在每个新 chunk 生成前调用，所有帧 age +1。"""
        for frame in self.frames:
            frame.age += 1

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_key_state: Tensor,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Tensor]]:
        """按 cosine similarity 检索最相关的 top-k 帧，返回结构化结果。

        Args:
            query_key_state: [dim] 当前 chunk 的查询嵌入
            top_k:           返回帧数，None 表示返回全部
            device:          输出 tensor 放置的设备，None 时跟随 query

        Returns:
            None if bank is empty, otherwise dict:
                key_states:    [k, dim]
                value_visuals: [k, dim]  — 传入 MemoryCrossAttention 的 KV
                timesteps:     [k]
                surprises:     [k]
                similarities:  [k]       — 与 query 的 cosine similarity
        """
        if not self.frames:
            return None

        device = device or query_key_state.device
        k = min(top_k, len(self.frames)) if top_k is not None else len(self.frames)

        all_key_states = torch.stack(
            [f.key_state for f in self.frames]
        ).to(device)  # [N, dim]

        # cosine similarity: [N]
        sims = F.cosine_similarity(
            query_key_state.unsqueeze(0).to(device),  # [1, dim]
            all_key_states,                            # [N, dim]
            dim=-1,
        )

        topk_sims, indices = torch.topk(sims, k=k)

        idx_list = indices.tolist()
        result = {
            "key_states": torch.stack(
                [self.frames[i].key_state for i in idx_list]
            ).to(device),  # [k, dim]
            "value_visuals": torch.stack(
                [self.frames[i].value_visual for i in idx_list]
            ).to(device),  # [k, dim]
            "timesteps": torch.tensor(
                [self.frames[i].timestep for i in idx_list],
                device=device, dtype=torch.long,
            ),  # [k]
            "surprises": torch.tensor(
                [self.frames[i].surprise_score for i in idx_list],
                device=device, dtype=torch.float32,
            ),  # [k]
            "similarities": topk_sims,  # [k]
        }

        logger.debug(
            "MemoryBank.retrieve: top-%d, sim_mean=%.4f, sim_std=%.4f",
            k, topk_sims.mean().item(),
            topk_sims.std().item() if k > 1 else 0.0,
        )

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_value_visuals(
        self, device: Optional[torch.device] = None
    ) -> Optional[Tensor]:
        """返回所有存储帧的 value_visual，[K, dim]。

        适用于不需要检索筛选、直接把全部 memory 传入 cross-attention 的场景。
        若 bank 为空返回 None。
        """
        if not self.frames:
            return None
        device = device or self.frames[0].value_visual.device
        return torch.stack([f.value_visual for f in self.frames]).to(device)

    def get_stats(self) -> Dict[str, float]:
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

    def size(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        """清空 Memory Bank（换 episode 时调用）。"""
        self.frames.clear()
        self.store_count = 0
        self.reject_count = 0
        self.evict_count = 0
        logger.info("MemoryBank: cleared.")

    def __repr__(self) -> str:
        scores = [f"{f.surprise_score:.3f}" for f in self.frames]
        return (
            f"MemoryBank(size={len(self.frames)}/{self.max_size}, "
            f"surprise=[{', '.join(scores)}])"
        )
