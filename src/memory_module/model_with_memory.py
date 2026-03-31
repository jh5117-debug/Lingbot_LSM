"""
model_with_memory.py — WanModel with Surprise-Driven Memory

在不修改 lingbot-world 原始代码的前提下，通过继承和包装引入 Memory 机制：
  - MemoryBlockWrapper: 包裹现有 WanAttentionBlock，在其输出后追加 Memory Cross-Attention
  - WanModelWithMemory: 继承 WanModel，将指定 blocks 替换为 MemoryBlockWrapper，
                        并添加 NFPHead；通过 dit_cond_dict 传递 memory_states

插入位置：每个被包裹的 WanAttentionBlock 完整输出之后（含 FFN），
          追加 memory_norm → MemoryCrossAttention → 残差连接。

此处对应 experiment_design.md Step 2（memory_attention.py）和 Step 3（model.py 修改）。

参考：
  - lingbot-world: wan/modules/model.py  WanModel, WanAttentionBlock, WanLayerNorm
  - WorldMem:      algorithms/worldmem/models/dit.py  memory attention 插入方式（残差+gate）
"""

import logging
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from torch import Tensor
from einops import rearrange

# ---- 引入 lingbot-world 模块 ----
_LINGBOT_WORLD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'refs', 'lingbot-world'
)
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

from wan.modules.model import WanModel, WanAttentionBlock, WanLayerNorm  # noqa: E402

from .memory_attention import MemoryCrossAttention  # noqa: E402
from .nfp_head import NFPHead  # noqa: E402

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# dit_cond_dict 中 memory_states 的 key（避免字符串 typo）
_MEMORY_STATES_KEY = "memory_states"


# ---------------------------------------------------------------------------
# MemoryBlockWrapper
# ---------------------------------------------------------------------------

class MemoryBlockWrapper(nn.Module):
    """包裹一个 WanAttentionBlock，在其输出后追加 Memory Cross-Attention。

    Forward 流程：
        1. block(x, **kwargs)          原始 Self-Attn + Camera FiLM + Text Cross-Attn + FFN
        2. memory_norm(x)              LayerNorm
        3. memory_cross_attn(x, M)     Memory Cross-Attention（Query=x，KV=memory_states M）
        4. x = x + output              残差连接

    当 dit_cond_dict 中不含 'memory_states' 时，跳过步骤 2-4，行为与原始 block 完全一致。

    Args:
        block:      待包裹的 WanAttentionBlock 实例（直接复用，不复制）
        dim:        模型隐藏维度
        num_heads:  注意力头数
        qk_norm:    是否对 Q/K 做 RMSNorm
        eps:        归一化 epsilon
    """

    def __init__(
        self,
        block: WanAttentionBlock,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.block = block
        self.memory_norm = WanLayerNorm(dim, eps)
        self.memory_cross_attn = MemoryCrossAttention(
            dim=dim, num_heads=num_heads, qk_norm=qk_norm, eps=eps
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Args:
            x:      [B, L, dim]
            **kwargs: 传递给内部 block 的全部参数
                      (e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict)

        Returns:
            x: [B, L, dim]  含记忆注入后的隐藏状态
        """
        # Step 1: 原始 block
        x = self.block(x, **kwargs)

        # Step 2-4: Memory Cross-Attention（仅当 memory_states 存在时）
        dit_cond_dict = kwargs.get("dit_cond_dict", None)
        if dit_cond_dict is not None and _MEMORY_STATES_KEY in dit_cond_dict:
            memory_states = dit_cond_dict[_MEMORY_STATES_KEY]  # [B, K, dim]
            x = x + self.memory_cross_attn(self.memory_norm(x), memory_states)

        return x


# ---------------------------------------------------------------------------
# WanModelWithMemory
# ---------------------------------------------------------------------------

class WanModelWithMemory(WanModel):
    """继承 WanModel，为指定 blocks 添加 Memory Cross-Attention，并增加 NFPHead。

    使用方式：
        # 从预训练权重加载原始模型
        base_model = WanModel.from_pretrained(ckpt_dir)

        # 转换为带记忆的版本（新增参数随机初始化）
        model = WanModelWithMemory.from_wan_model(
            base_model,
            memory_layers=None,   # None = 全部 blocks
            max_memory_size=8,
        )

        # 推理时传入 memory_states
        output = model(
            x, t, context, seq_len,
            dit_cond_dict={"c2ws_plucker_emb": ...},
            memory_states=memory_bank_states,  # [1, K, dim]
        )

    Args:
        memory_layers:   要插入 Memory Cross-Attention 的 block 索引列表。
                         None = 全部 blocks（默认）。
                         建议从后半段 blocks 开始，例如 range(20, 40)。
        max_memory_size: Memory Bank 最大容量 K，用于初始化文档，不影响模型权重。
        其余参数与 WanModel 完全相同。
    """

    def __init__(
        self,
        *args,
        memory_layers: Optional[List[int]] = None,
        max_memory_size: int = 8,
        **kwargs,
    ):
        # 先调用 WanModel.__init__，创建原始 blocks
        super().__init__(*args, **kwargs)

        # 确定要包裹的 block 索引
        if memory_layers is None:
            memory_layers = list(range(len(self.blocks)))
        self._memory_layers = memory_layers
        self._max_memory_size = max_memory_size

        # 将指定 blocks 替换为 MemoryBlockWrapper
        for i in memory_layers:
            self.blocks[i] = MemoryBlockWrapper(
                block=self.blocks[i],
                dim=self.dim,
                num_heads=self.num_heads,
                qk_norm=self.qk_norm,
                eps=self.eps,
            )

        # NFPHead：预测下一帧 latent，用于 Surprise score 计算
        self.nfp_head = NFPHead(dim=self.dim, z_dim=self.out_dim)

        logger.info(
            "WanModelWithMemory: wrapped %d blocks with MemoryBlockWrapper "
            "(layers=%s), added NFPHead(dim=%d, z_dim=%d).",
            len(memory_layers), memory_layers, self.dim, self.out_dim,
        )

    # ------------------------------------------------------------------
    # forward：注入 memory_states
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        memory_states: Optional[Tensor] = None,
    ):
        """在原始 WanModel.forward 基础上支持 memory_states 注入。

        Args:
            memory_states: [B, K, dim]  Memory Bank 检索到的历史帧 pose_emb。
                           若为 None，行为与原始 WanModel 完全一致。
            其余参数见 WanModel.forward 文档。

        Returns:
            与 WanModel.forward 相同：List[Tensor]，每项 [C_out, F, H/8, W/8]
        """
        if memory_states is not None:
            # 注入到 dit_cond_dict，MemoryBlockWrapper 会从中取出
            dit_cond_dict = dict(dit_cond_dict) if dit_cond_dict is not None else {}
            dit_cond_dict[_MEMORY_STATES_KEY] = memory_states

        return super().forward(
            x, t, context, seq_len, y=y, dit_cond_dict=dit_cond_dict
        )

    # ------------------------------------------------------------------
    # 工厂方法：从预训练 WanModel 转换
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 推理辅助：提取 per-frame pose embedding（用于 MemoryBank 存储与检索）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_projected_frame_embs(self, c2ws_plucker_emb: Tensor) -> Tensor:
        """将 raw plucker embedding 投影到模型空间，返回 per-frame 嵌入。

        镜像 WanModel.forward() 内部的 camera 处理逻辑，返回的嵌入与
        传入 WanAttentionBlock 的 c2ws_plucker_emb 保持同一向量空间，
        可直接存入 MemoryBank 并用于 cosine similarity 检索。

        Args:
            c2ws_plucker_emb: [1, C, lat_f, lat_h, lat_w]
                              与 dit_cond_dict["c2ws_plucker_emb"] 进 model.forward() 之前的
                              原始张量相同（chunk 之前）

        Returns:
            frame_embs: [lat_f, dim]  每帧经过均值池化的模型空间 pose 嵌入
        """
        _, _C, lat_f, lat_h, lat_w = c2ws_plucker_emb.shape
        p_t, p_h, p_w = self.patch_size  # (1, 2, 2)
        h_p = lat_h // p_h              # spatial patch 格数（height）
        w_p = lat_w // p_w              # spatial patch 格数（width）

        # 与 WanModel.forward() 完全相同的 rearrange
        x = rearrange(
            c2ws_plucker_emb,
            '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
            c1=p_t, c2=p_h, c3=p_w,
        )  # [1, lat_f * h_p * w_p, raw_dim]

        x = self.patch_embedding_wancamctrl(x)  # [1, L, dim]
        hidden = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(x))
        )
        projected = x + hidden  # [1, L, dim]

        # Mean-pool over spatial patches per frame → [lat_f, dim]
        projected = projected.view(1, lat_f, h_p * w_p, self.dim)
        frame_embs = projected.mean(dim=2).squeeze(0)
        return frame_embs

    @classmethod
    def from_wan_model(
        cls,
        base_model: WanModel,
        memory_layers: Optional[List[int]] = None,
        max_memory_size: int = 8,
    ) -> "WanModelWithMemory":
        """从已加载的预训练 WanModel 转换为 WanModelWithMemory。

        新增的参数（MemoryBlockWrapper 中的 memory_cross_attn 和 nfp_head）
        将随机初始化，需要通过微调来学习。

        Args:
            base_model:      已加载预训练权重的 WanModel 实例
            memory_layers:   要插入记忆注意力的 block 索引，None = 全部
            max_memory_size: Memory Bank 容量 K

        Returns:
            WanModelWithMemory 实例，原有权重不变，新增参数随机初始化
        """
        cfg = base_model.config
        model = cls(
            model_type=cfg.model_type,
            control_type=getattr(cfg, 'control_type', 'cam'),
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=cfg.in_dim,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            text_dim=cfg.text_dim,
            out_dim=cfg.out_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
            memory_layers=memory_layers,
            max_memory_size=max_memory_size,
        )

        # 加载原始权重（MemoryBlockWrapper 内的 block 权重也被正确加载）
        missing, unexpected = model.load_state_dict(
            base_model.state_dict(), strict=False
        )
        if unexpected:
            logger.warning("from_wan_model: unexpected keys: %s", unexpected)
        if missing:
            logger.info(
                "from_wan_model: %d keys not in base model "
                "(expected: memory_cross_attn + nfp_head): %s",
                len(missing), missing[:5],
            )

        return model
