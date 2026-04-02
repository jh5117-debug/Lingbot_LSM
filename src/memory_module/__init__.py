"""
memory_module -- Surprise-Driven Memory Bank for LingBot-World (v2)

New modules, no modifications to lingbot-world source.

Core components:
    MemoryBank          -- Store / retrieve historical frames (surprise-driven eviction)
    MemoryFrame         -- Single frame memory data structure
    MemoryCrossAttention -- Historical frame Cross-Attention (Query=current, KV=history, gated)
    NFPHead             -- Next Frame Prediction Head (per-frame surprise source)
    WanModelWithMemory  -- Memory-enhanced WanModel (inheritance, no source modification)
    MemoryBlockWrapper  -- Single block wrapper (WanAttentionBlock + MemoryCrossAttention)
"""

from .memory_bank import MemoryBank, MemoryFrame
from .nfp_head import NFPHead

__all__ = ['MemoryBank', 'MemoryFrame', 'NFPHead']

try:
    from .memory_attention import MemoryCrossAttention, RMSNorm
    from .model_with_memory import MemoryBlockWrapper, WanModelWithMemory
    __all__ += ['MemoryCrossAttention', 'RMSNorm', 'MemoryBlockWrapper', 'WanModelWithMemory']
except (ImportError, RuntimeError):
    # CUDA unavailable (e.g. login node): skip, import submodules directly as needed
    pass
