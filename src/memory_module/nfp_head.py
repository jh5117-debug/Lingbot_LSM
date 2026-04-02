"""
nfp_head.py -- Next Frame Prediction Head (v2: per-frame)

2-layer MLP: dim (5120) -> dim (5120) -> z_dim (16)
Supports both clip-level (backward compat) and per-frame prediction.

v2 changes (MEMORY_FIX_REVIEW_AND_WANDB_PLAN.md Task E):
  - New forward_per_frame(): reshapes hidden_states to [B, lat_f, spatial, dim],
    does per-frame spatial mean-pool, outputs [B, lat_f, z_dim]
  - compute_surprise_per_frame(): returns [B, lat_f]
  - compute_loss_per_frame(): target is next-frame latent spatial mean,
    mask excludes last frame

References:
  - cambrian-s: cambrian/model/cambrian_arch.py NFP head (2-layer MLP, L200-205)
  - cambrian-s: cambrian/model/language_model/cambrian_qwen2.py nfp_loss
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
    """Next Frame Prediction Head (v2: per-frame support).

    Accepts hidden states from the last transformer block,
    predicts per-frame VAE latent (spatial mean-pooled),
    used to compute per-frame Surprise scores for Memory Bank.

    Network (ref cambrian-s):
        hidden_states [B, L, dim]
            -> reshape to [B, lat_f, spatial_tokens, dim]
            -> mean_pool over spatial per frame -> [B, lat_f, dim]
            -> Linear(dim, dim) + GELU
            -> Linear(dim, z_dim)
            -> pred_latent [B, lat_f, z_dim]

    Args:
        dim:   model hidden dim (A14B: 5120)
        z_dim: VAE latent channels (lingbot-world: 16)
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

        # Init (ref cambrian-s trunc_normal)
        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.constant_(self.mlp[0].bias, 0.0)
        nn.init.trunc_normal_(self.mlp[2].weight, std=0.02)
        nn.init.constant_(self.mlp[2].bias, 0.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Clip-level prediction (backward compat).

        Args:
            hidden_states: [B, L, dim]

        Returns:
            pred_latent: [B, z_dim]
        """
        x = hidden_states.mean(dim=1)  # [B, dim]
        return self.mlp(x)             # [B, z_dim]

    def forward_per_frame(
        self,
        hidden_states: Tensor,
        lat_f: int,
        num_spatial_per_frame: int,
    ) -> Tensor:
        """Per-frame prediction (v2).

        Args:
            hidden_states:          [B, L, dim]  L = lat_f * num_spatial_per_frame
            lat_f:                  number of latent frames
            num_spatial_per_frame:  number of spatial tokens per frame
                                    = lat_h * lat_w / (patch_h * patch_w)

        Returns:
            pred_latent: [B, lat_f, z_dim]  per-frame prediction
        """
        B = hidden_states.shape[0]
        # Reshape: [B, lat_f, spatial, dim]
        hs = hidden_states.view(B, lat_f, num_spatial_per_frame, self.dim)
        # Per-frame spatial mean-pool: [B, lat_f, dim]
        hs_pooled = hs.mean(dim=2)
        # MLP: [B, lat_f, z_dim]
        return self.mlp(hs_pooled)

    # ------------------------------------------------------------------
    # Surprise Score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_surprise(pred_latent: Tensor, actual_latent: Tensor) -> Tensor:
        """Clip-level surprise (backward compat).

        Args:
            pred_latent:   [B, z_dim]
            actual_latent: [B, z_dim]

        Returns:
            surprise: [B]  range [0, 2]
        """
        cos_sim = F.cosine_similarity(pred_latent, actual_latent, dim=-1)
        return 1.0 - cos_sim

    @staticmethod
    def compute_surprise_per_frame(
        pred_latent: Tensor,
        actual_latent: Tensor,
    ) -> Tensor:
        """Per-frame surprise (v2).

        Args:
            pred_latent:   [B, lat_f, z_dim]
            actual_latent: [B, lat_f, z_dim]

        Returns:
            surprise: [B, lat_f]  range [0, 2]
        """
        cos_sim = F.cosine_similarity(pred_latent, actual_latent, dim=-1)  # [B, lat_f]
        return 1.0 - cos_sim

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
        """Clip-level NFP loss (backward compat).

        Args:
            pred_latent:   [B, z_dim]
            target_latent: [B, z_dim]
            loss_mask:     [B]
            mse_weight:    MSE loss weight (default 0.1)
            cosine_weight: Cosine loss weight (default 0.1)

        Returns:
            dict: 'mse', 'cosine', 'total'
        """
        B = pred_latent.size(0)
        ones = torch.ones(B, device=pred_latent.device, dtype=pred_latent.dtype)

        mse_per_sample = F.mse_loss(pred_latent, target_latent, reduction='none').mean(-1)
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

    @staticmethod
    def compute_loss_per_frame(
        pred_latent: Tensor,
        video_latent: Tensor,
        mse_weight: float = 0.1,
        cosine_weight: float = 0.1,
    ) -> dict:
        """Per-frame NFP loss (v2).

        Target: next frame's latent spatial mean.
        Mask: exclude last frame (no next-frame target).

        Args:
            pred_latent:  [B, lat_f, z_dim]  NFPHead per-frame prediction
            video_latent: [z_dim, lat_f, lat_h, lat_w]  original VAE latent (single sample)

        Returns:
            dict: 'mse', 'cosine', 'total', 'per_frame_surprise' [lat_f-1]
        """
        # Build per-frame target: next frame's spatial mean
        # video_latent: [z_dim, lat_f, lat_h, lat_w]
        # target for frame t = video_latent[:, t+1].mean(dim=(-2,-1))  -> [z_dim]
        lat_f = video_latent.shape[1]
        z_dim = video_latent.shape[0]

        # [lat_f-1, z_dim]: target for frame 0..lat_f-2 is frame 1..lat_f-1
        target_frames = video_latent[:, 1:].mean(dim=(-2, -1)).T  # [lat_f-1, z_dim]
        target_frames = target_frames.unsqueeze(0)  # [1, lat_f-1, z_dim]

        # Trim pred to match (exclude last frame prediction)
        pred_trimmed = pred_latent[:, :lat_f - 1]  # [B, lat_f-1, z_dim]

        B, T, D = pred_trimmed.shape
        pred_flat = pred_trimmed.reshape(B * T, D)
        target_flat = target_frames.expand(B, -1, -1).reshape(B * T, D)

        # MSE
        mse_per = F.mse_loss(pred_flat, target_flat, reduction='none').mean(-1)  # [B*T]
        mse_loss = mse_per.mean() * mse_weight

        # Cosine embedding loss
        ones = torch.ones(B * T, device=pred_flat.device, dtype=pred_flat.dtype)
        cosine_per = F.cosine_embedding_loss(pred_flat, target_flat, ones, reduction='none')
        cosine_loss = cosine_per.mean() * cosine_weight

        # Per-frame surprise for memory bank
        with torch.no_grad():
            cos_sim = F.cosine_similarity(pred_trimmed, target_frames.expand(B, -1, -1), dim=-1)
            per_frame_surprise = (1.0 - cos_sim).squeeze(0)  # [lat_f-1]

        return {
            'mse': mse_loss,
            'cosine': cosine_loss,
            'total': mse_loss + cosine_loss,
            'per_frame_surprise': per_frame_surprise,  # [lat_f-1]
        }
