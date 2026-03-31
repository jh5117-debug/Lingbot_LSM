"""
nfp_head.py — Next Frame Prediction Head

2层 MLP：dim (5120) → dim (5120) → z_dim (16)
预测当前帧对应的 VAE latent 均值，计算 per-frame Surprise score。

Surprise = 1 - cosine_similarity(pred_latent, actual_latent)
  - 值域 [0, 2]；接近 0 表示"意料之中"，接近 2 表示"非常意外"
  - 训练时同时用 MSE + Cosine loss（权重各 0.1，参考 cambrian-s）

参考：
  - cambrian-s: cambrian/model/cambrian_arch.py NFP head 结构（2-layer MLP）
  - cambrian-s: cambrian/model/language_model/cambrian_qwen2.py nfp_loss（MSE + Cosine）
  - 论文设计：surprise = 1 - cosine_sim（Cambrian-S Section 3.2）
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class NFPHead(nn.Module):
    """Next Frame Prediction Head。

    接受模型最后一个 transformer block 的输出隐藏状态，
    预测下一帧的 VAE latent（spatial mean-pooled），
    用于计算 Surprise score 来筛选哪些帧值得存入 Memory Bank。

    网络结构（参考 cambrian-s）：
        hidden_states [B, L, dim]
            → mean_pool over L → [B, dim]
            → Linear(dim, dim) + GELU
            → Linear(dim, z_dim)
            → pred_latent [B, z_dim]

    Args:
        dim:   模型隐藏维度（A14B 配置为 5120）
        z_dim: VAE latent 通道数（lingbot-world 配置为 16）
    """

    def __init__(self, dim: int = 5120, z_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, z_dim),
        )

        # 初始化（参考 cambrian-s trunc_normal）
        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.constant_(self.mlp[0].bias, 0.0)
        nn.init.trunc_normal_(self.mlp[2].weight, std=0.02)
        nn.init.constant_(self.mlp[2].bias, 0.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """预测下一帧的 VAE latent（spatial mean-pool 后的通道向量）。

        Args:
            hidden_states: [B, L, dim]  transformer 最后一层的输出

        Returns:
            pred_latent: [B, z_dim]  预测的下一帧 VAE latent（mean over spatial）
        """
        # Mean-pool over sequence length → [B, dim]
        x = hidden_states.mean(dim=1)
        # MLP → [B, z_dim]
        return self.mlp(x)

    # ------------------------------------------------------------------
    # Surprise Score 计算
    # ------------------------------------------------------------------

    @staticmethod
    def compute_surprise(pred_latent: Tensor, actual_latent: Tensor) -> Tensor:
        """计算 per-frame Surprise score。

        Surprise = 1 - cosine_similarity(pred, actual)
        值域 [0, 2]，越大说明该帧越"出乎意料"，越值得存入 Memory Bank。

        Args:
            pred_latent:   [B, z_dim]  NFPHead 预测的下一帧 latent
            actual_latent: [B, z_dim]  实际下一帧经过 mean-pool 的 VAE latent

        Returns:
            surprise: [B]  per-sample Surprise score
        """
        cos_sim = F.cosine_similarity(pred_latent, actual_latent, dim=-1)  # [B]
        return 1.0 - cos_sim  # [B], 值域 [0, 2]

    # ------------------------------------------------------------------
    # Training Loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        pred_latent: Tensor,
        target_latent: Tensor,
        loss_mask: Tensor = None,
        mse_weight: float = 0.1,
        cosine_weight: float = 0.1,
    ) -> dict:
        """计算 NFP 训练 loss（MSE + Cosine，参考 cambrian-s nfp_loss）。

        Args:
            pred_latent:   [B, z_dim]  NFPHead 的预测输出
            target_latent: [B, z_dim]  目标下一帧 latent（来自 VAE encoder）
            loss_mask:     [B]         有效帧的掩码（1=计算 loss，0=跳过）
                                       None 时所有帧均参与
            mse_weight:    MSE loss 的全局权重（默认 0.1）
            cosine_weight: Cosine loss 的全局权重（默认 0.1）

        Returns:
            dict with keys:
                'mse':    加权后的 MSE loss（scalar）
                'cosine': 加权后的 Cosine loss（scalar）
                'total':  mse + cosine（scalar，加到主 diffusion loss 上）
        """
        B = pred_latent.size(0)
        ones = torch.ones(B, device=pred_latent.device, dtype=pred_latent.dtype)

        # MSE loss: per-sample mean over z_dim → [B]
        mse_per_sample = F.mse_loss(pred_latent, target_latent, reduction='none').mean(-1)

        # Cosine embedding loss: 1 - cos_sim → [B]
        cosine_per_sample = F.cosine_embedding_loss(
            pred_latent, target_latent, ones, reduction='none'
        )

        if loss_mask is not None:
            denom = loss_mask.float().sum().clamp(min=1e-12)
            mse_loss = (mse_per_sample * loss_mask.float()).sum() / denom
            cosine_loss = (cosine_per_sample * loss_mask.float()).sum() / denom
        else:
            mse_loss = mse_per_sample.mean()
            cosine_loss = cosine_per_sample.mean()

        mse_loss = mse_loss * mse_weight
        cosine_loss = cosine_loss * cosine_weight

        return {
            'mse': mse_loss,
            'cosine': cosine_loss,
            'total': mse_loss + cosine_loss,
        }
