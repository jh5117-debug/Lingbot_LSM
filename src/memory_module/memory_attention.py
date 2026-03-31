"""
memory_attention.py — Memory Cross-Attention Module

结构参考 lingbot-world 的 WanCrossAttention，Query 来自当前帧特征，
Key/Value 来自 MemoryBank 检索到的历史帧 pose_emb。

与 WanCrossAttention 的区别：
  - Key/Value 是 memory_states [B, K, dim]，而非文本嵌入
  - 不使用 RoPE（与 WanCrossAttention 一致）
  - 自带 RMSNorm，不依赖 lingbot-world 内部类

参考：
  - lingbot-world: wan/modules/model.py WanCrossAttention（接口风格）
  - lingbot-world: wan/modules/attention.py flash_attention（底层计算）
"""

import logging
import os
import sys

import torch
import torch.nn as nn
from torch import Tensor

# ---- sys.path（供 forward 内懒加载 flash_attention 使用）----
_LINGBOT_WORLD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'refs', 'lingbot-world'
)
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

# flash_attention 已移至 MemoryCrossAttention.forward() 内懒加载，
# 避免模块导入时触发 wan/__init__.py → T5EncoderModel → torch.cuda.current_device()

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """轻量 RMSNorm，不依赖 lingbot-world 内部类。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # 在 float32 下计算，避免 bfloat16 精度问题
        x_f32 = x.float()
        norm = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * norm * self.weight).to(x.dtype)


class MemoryCrossAttention(nn.Module):
    """历史帧 Memory Cross-Attention。

    Query 来自当前帧的隐藏状态 x，Key/Value 来自 Memory Bank 检索到的历史帧。
    结构与 WanCrossAttention 保持一致（无 RoPE，支持 Flash Attention）。

    Args:
        dim:       模型隐藏维度（A14B 配置为 5120）
        num_heads: 注意力头数（A14B 配置为 40）
        qk_norm:   是否对 Q/K 做 RMSNorm，默认 True（与 lingbot-world 一致）
        eps:       归一化 epsilon
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        # Q/K 归一化（与 WanSelfAttention 保持一致）
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        memory_states: Tensor,
        memory_lens: Tensor = None,
    ) -> Tensor:
        """
        Args:
            x:             [B, L, dim]  当前帧序列（Query 来源）
            memory_states: [B, K, dim]  Memory Bank 检索到的历史帧 pose_emb
            memory_lens:   [B]          每个样本实际有效的 memory 帧数（用于 padding mask）
                                        若所有样本 memory 数相同可传 None

        Returns:
            out: [B, L, dim]  memory cross-attention 的输出（残差加法前）
        """
        from wan.modules.attention import flash_attention  # lazy import：仅在 forward 调用时加载 wan

        # dtype 对齐：直接读 Linear weight dtype，避免 next(parameters()) 在 ZeRO-3 下不可靠
        target_dtype = self.q.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if memory_states is not None and memory_states.dtype != target_dtype:
            memory_states = memory_states.to(target_dtype)

        B, L, _ = x.shape
        if memory_states is None:
            return x.new_zeros(B, L, self.dim)
        K = memory_states.shape[1]

        # Projection + QK-norm（在 view 之前 norm，与 WanCrossAttention 一致）
        q = self.norm_q(self.q(x)).view(B, L, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(memory_states)).view(B, K, self.num_heads, self.head_dim)
        v = self.v(memory_states).view(B, K, self.num_heads, self.head_dim)

        # Flash Attention: [B, L, num_heads, head_dim]
        out = flash_attention(q, k, v, k_lens=memory_lens)

        # Merge heads: [B, L, dim]
        out = self.o(out.flatten(2))
        return out
