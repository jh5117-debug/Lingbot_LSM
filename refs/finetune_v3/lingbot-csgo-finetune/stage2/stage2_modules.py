"""
Stage 2 modules for multi-view CSGO video generation.

Adds BEV cross-attention and visibility-sparse cross-player attention
to the frozen Stage 1 WanModel. Both modules use zero-initialized gates
for progressive training (ShareVerse-style).

Architecture per DiT block:
  1. Self-Attention (frozen)
  2. Camera Injection (frozen)
  3. BEV Cross-Attention (NEW, trainable) — gate_bev * bev_cross_attn(x, bev_tokens)
  4. Cross-Player Attention (NEW, trainable) — gate_xp * cross_player_attn(x, visible_features)
  5. Text Cross-Attention (frozen)
  6. FFN (frozen)
"""

import math
import logging
from functools import wraps
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# BEV Encoder: CNN that converts BEV feature maps to token sequences
# ---------------------------------------------------------------------------

class BEVEncoder(nn.Module):
    """
    Encodes a multi-channel BEV map into a sequence of tokens compatible
    with WanModel's hidden dimension.

    Input:  [B, C_bev, H_bev, W_bev]  (e.g., [B, 7, 256, 256])
    Output: [B, N_tokens, dim]         (e.g., [B, 256, 2048])

    Uses a lightweight CNN to downsample spatially, then projects to model dim.
    """

    def __init__(
        self,
        in_channels: int = 7,
        dim: int = 2048,
        hidden_channels: int = 128,
        bev_size: int = 256,
        token_grid: int = 16,
    ):
        """
        Args:
            in_channels: Number of BEV channels (static + dynamic).
            dim: WanModel hidden dimension (must match).
            hidden_channels: CNN intermediate channels.
            bev_size: Input BEV spatial size (assumed square).
            token_grid: Output spatial grid size (tokens = token_grid^2).
        """
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.token_grid = token_grid
        self.num_tokens = token_grid * token_grid

        # Number of 2x downsampling stages needed
        num_stages = int(math.log2(bev_size // token_grid))
        assert 2 ** num_stages == bev_size // token_grid, \
            f"bev_size/token_grid must be power of 2, got {bev_size}/{token_grid}"

        layers = []
        ch_in = in_channels
        for i in range(num_stages):
            ch_out = hidden_channels * (2 ** min(i, 2))  # cap at 512
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.GroupNorm(min(32, ch_out), ch_out),
                nn.GELU(),
            ])
            ch_in = ch_out

        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(ch_in, dim)

        # Learnable position embedding for BEV tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, C_bev, H_bev, W_bev]
        Returns:
            tokens: [B, N_tokens, dim]
        """
        x = self.cnn(bev)  # [B, ch_out, token_grid, token_grid]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, N_tokens, ch_out]
        x = self.proj(x)  # [B, N_tokens, dim]
        x = x + self.pos_embed
        return x


# ---------------------------------------------------------------------------
# BEV Cross-Attention: query BEV tokens from video features
# ---------------------------------------------------------------------------

class BEVCrossAttention(nn.Module):
    """
    Cross-attention where video tokens (query) attend to BEV tokens (key/value).
    Uses the same QK-norm + flash attention pattern as WanCrossAttention.
    """

    def __init__(self, dim: int = 2048, bottleneck_dim: int = 512,
                 num_heads: int = 8, qk_norm: bool = True, eps: float = 1e-6):
        """
        Uses a bottleneck projection to keep parameter count manageable.
        Instead of full dim→dim attention (104M params/block at dim=5120),
        projects down to bottleneck_dim first (~10M params/block at bottleneck=512).
        """
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.head_dim = bottleneck_dim // num_heads

        # Bottleneck projections: dim → bottleneck → dim
        self.q = nn.Linear(dim, bottleneck_dim)
        self.k = nn.Linear(dim, bottleneck_dim)
        self.v = nn.Linear(dim, bottleneck_dim)
        self.o = nn.Linear(bottleneck_dim, dim)

        if qk_norm:
            self.norm_q = nn.RMSNorm(bottleneck_dim, eps=eps)
            self.norm_k = nn.RMSNorm(bottleneck_dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.norm_in = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q, self.k, self.v]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(
        self,
        x: torch.Tensor,
        bev_tokens: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Video features [B, L, C]
            bev_tokens: BEV token features [B, N_bev, C]
        Returns:
            out: [B, L, C]
        """
        b, l, c = x.shape
        n, d = self.num_heads, self.head_dim
        n_bev = bev_tokens.shape[1]

        x_normed = self.norm_in(x.float()).type_as(x)

        q = self.norm_q(self.q(x_normed)).view(b, l, n, d)
        k = self.norm_k(self.k(bev_tokens)).view(b, n_bev, n, d)
        v = self.v(bev_tokens).view(b, n_bev, n, d)

        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.transpose(1, 2).to(torch.bfloat16)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().flatten(2)
        out = self.o(out.type_as(x))
        return out


# ---------------------------------------------------------------------------
# Cross-Player Attention: sparse cross-attention between visible players
# ---------------------------------------------------------------------------

class CrossPlayerAttention(nn.Module):
    """
    Cross-attention where player i's video tokens attend to visible players'
    video features. Only computes attention for actually visible player pairs.
    """

    def __init__(self, dim: int = 2048, bottleneck_dim: int = 512,
                 num_heads: int = 8, qk_norm: bool = True, eps: float = 1e-6):
        """Bottleneck cross-attention, same design as BEVCrossAttention."""
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.head_dim = bottleneck_dim // num_heads

        self.q = nn.Linear(dim, bottleneck_dim)
        self.k = nn.Linear(dim, bottleneck_dim)
        self.v = nn.Linear(dim, bottleneck_dim)
        self.o = nn.Linear(bottleneck_dim, dim)

        if qk_norm:
            self.norm_q = nn.RMSNorm(bottleneck_dim, eps=eps)
            self.norm_k = nn.RMSNorm(bottleneck_dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.norm_in = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q, self.k, self.v]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(
        self,
        x: torch.Tensor,
        visible_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Current player's features [B, L, C]
            visible_features: Concatenated features of visible players [B, L_vis, C]
                              L_vis = sum of sequence lengths of all visible players
        Returns:
            out: [B, L, C]
        """
        b, l, c = x.shape
        n, d = self.num_heads, self.head_dim
        l_vis = visible_features.shape[1]

        x_normed = self.norm_in(x.float()).type_as(x)

        q = self.norm_q(self.q(x_normed)).view(b, l, n, d)
        k = self.norm_k(self.k(visible_features)).view(b, l_vis, n, d)
        v = self.v(visible_features).view(b, l_vis, n, d)

        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.transpose(1, 2).to(torch.bfloat16)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().flatten(2)
        out = self.o(out.type_as(x))
        return out


# ---------------------------------------------------------------------------
# Per-block Stage 2 adapters (BEV cross-attn + cross-player attn + gates)
# ---------------------------------------------------------------------------

class Stage2BlockAdapter(nn.Module):
    """
    Adapter module for a single WanAttentionBlock.
    Contains BEV cross-attention, optionally cross-player attention, and their gates.
    """

    def __init__(self, dim: int = 2048, bottleneck_dim: int = 512,
                 num_heads: int = 8, eps: float = 1e-6,
                 enable_cross_player: bool = False):
        super().__init__()
        self.bev_cross_attn = BEVCrossAttention(dim, bottleneck_dim, num_heads, eps=eps)
        self.gate_bev = nn.Parameter(torch.zeros(1, 1, dim))

        # Only create cross-player modules if needed (Phase 2b)
        self.has_cross_player = enable_cross_player
        if enable_cross_player:
            self.cross_player_attn = CrossPlayerAttention(dim, bottleneck_dim, num_heads, eps=eps)
            self.gate_cross_player = nn.Parameter(torch.zeros(1, 1, dim))

    def forward_bev(self, x: torch.Tensor, bev_tokens: torch.Tensor) -> torch.Tensor:
        """Apply gated BEV cross-attention."""
        return self.gate_bev * self.bev_cross_attn(x, bev_tokens)

    def forward_cross_player(self, x: torch.Tensor, visible_features: torch.Tensor) -> torch.Tensor:
        """Apply gated cross-player attention."""
        if not self.has_cross_player:
            return torch.zeros_like(x)
        return self.gate_cross_player * self.cross_player_attn(x, visible_features)


# ---------------------------------------------------------------------------
# Stage2ModelWrapper: wraps WanModel with Stage 2 modules
# ---------------------------------------------------------------------------

class Stage2ModelWrapper(nn.Module):
    """
    Wraps a frozen WanModel, adding per-block BEV cross-attention and
    cross-player attention modules. Only the new modules are trainable.

    Usage:
        base_model = WanModel.from_pretrained(...)
        wrapper = Stage2ModelWrapper(base_model, bev_channels=10)
        wrapper.freeze_base_model()

        # Forward pass:
        pred = wrapper(
            x=[noisy_latent],
            t=t,
            context=context,
            seq_len=seq_len,
            y=[y],
            dit_cond_dict=dit_cond_dict,
            bev_map=bev_map,                    # [B, C_bev, H_bev, W_bev]
            visible_player_features=vis_feats,  # optional
        )
    """

    def __init__(
        self,
        base_model: nn.Module,
        bev_channels: int = 10,
        bev_size: int = 256,
        bev_token_grid: int = 16,
        enable_cross_player: bool = False,
        adapter_stride: int = 4,
    ):
        """
        Args:
            base_model: Frozen WanModel from Stage 1.
            bev_channels: Number of channels in the BEV map (static + dynamic).
            bev_size: Spatial size of BEV map (square).
            bev_token_grid: Grid size for BEV tokens (tokens = grid^2).
            enable_cross_player: Whether to enable cross-player attention (Phase 2b).
            adapter_stride: Only add adapters every N blocks (saves memory).
                           stride=4 with 40 blocks → 10 adapted blocks.
        """
        super().__init__()
        self.base_model = base_model
        self.enable_cross_player = enable_cross_player

        dim = base_model.dim
        num_heads = base_model.num_heads
        num_layers = base_model.num_layers

        # BEV encoder (shared across all blocks)
        self.bev_encoder = BEVEncoder(
            in_channels=bev_channels,
            dim=dim,
            bev_size=bev_size,
            token_grid=bev_token_grid,
        )

        # Per-block adapters (only every adapter_stride blocks to save memory)
        # E.g., stride=4 with 40 blocks → adapters at blocks 0,4,8,...,36 (10 total)
        bottleneck_dim = 512
        self.adapter_stride = adapter_stride
        self.adapted_block_indices = list(range(0, num_layers, adapter_stride))
        self.block_adapters = nn.ModuleList([
            Stage2BlockAdapter(
                dim=dim, bottleneck_dim=bottleneck_dim,
                num_heads=8, enable_cross_player=enable_cross_player,
            )
            for _ in self.adapted_block_indices
        ])

        # Persistent storage for BEV/cross-player tokens (avoids gradient checkpoint issues)
        # These are set in forward() and read by patched block forwards
        self._current_bev_tokens = None
        self._current_vis_features = None

        # Install hooks to intercept block forward passes
        self._install_hooks()

        num_adapted = len(self.adapted_block_indices)
        logging.info(
            f"Stage2ModelWrapper: {num_adapted}/{num_layers} blocks adapted (stride={adapter_stride}), "
            f"BEV encoder ({bev_channels}ch → {bev_token_grid}x{bev_token_grid} tokens), "
            f"cross_player={'enabled' if enable_cross_player else 'disabled'}"
        )
        trainable = sum(p.numel() for p in self.trainable_parameters())
        logging.info(f"Stage 2 trainable parameters: {trainable:,}")

    def freeze_base_model(self):
        """Freeze all parameters in the base WanModel."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        frozen = sum(p.numel() for p in self.base_model.parameters())
        logging.info(f"Frozen base model: {frozen:,} parameters")

    def trainable_parameters(self):
        """Yield only Stage 2 trainable parameters."""
        for param in self.bev_encoder.parameters():
            yield param
        for adapter in self.block_adapters:
            for param in adapter.parameters():
                yield param

    def _install_hooks(self):
        """
        Use register_forward_hook to inject BEV and cross-player attention
        AFTER each block's forward. This preserves the original block.forward
        intact, which is critical for ZeRO-3 + gradient checkpointing compatibility.

        Architecture difference vs monkey-patch: BEV attention is applied after
        the full block (including FFN) rather than between cam injection and
        text cross-attention. Functionally equivalent due to zero-init gates.
        """
        self._hook_handles = []
        for adapter_idx, block_idx in enumerate(self.adapted_block_indices):
            block = self.base_model.blocks[block_idx]
            adapter = self.block_adapters[adapter_idx]

            def make_hook(adpt, wrapper):
                def hook_fn(module, input, output):
                    x = output

                    # BEV Cross-Attention
                    if wrapper._current_bev_tokens is not None:
                        x = x + adpt.forward_bev(x, wrapper._current_bev_tokens)

                    # Cross-Player Attention (Phase 2b only)
                    if wrapper._current_vis_features is not None:
                        x = x + adpt.forward_cross_player(x, wrapper._current_vis_features)

                    return x
                return hook_fn

            handle = block.register_forward_hook(make_hook(adapter, self))
            self._hook_handles.append(handle)

    def encode_bev(self, bev_map: torch.Tensor) -> torch.Tensor:
        """
        Encode BEV map into tokens.
        Args:
            bev_map: [B, C_bev, H_bev, W_bev]
        Returns:
            bev_tokens: [B, N_tokens, dim]
        """
        return self.bev_encoder(bev_map)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        bev_map: Optional[torch.Tensor] = None,
        visible_player_features: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the wrapped model.

        Args:
            x, t, context, seq_len, y: Same as WanModel.forward()
            dit_cond_dict: Camera control signals (from Stage 1).
            bev_map: [B, C_bev, H_bev, W_bev] BEV feature map.
            visible_player_features: [B, L_vis, C] concatenated features
                of visible players (for Phase 2b cross-player attention).
        Returns:
            Same as WanModel.forward()
        """
        # Store BEV tokens and vis features on wrapper (read by patched block forwards)
        # This avoids passing them through dit_cond_dict which breaks gradient checkpointing
        if bev_map is not None:
            self._current_bev_tokens = self.encode_bev(bev_map.to(torch.bfloat16))
        else:
            self._current_bev_tokens = None

        if visible_player_features is not None and self.enable_cross_player:
            self._current_vis_features = visible_player_features
        else:
            self._current_vis_features = None

        if dit_cond_dict is None:
            dit_cond_dict = {}

        return self.base_model(
            x=x, t=t, context=context,
            seq_len=seq_len, y=y,
            dit_cond_dict=dit_cond_dict,
        )

    def get_intermediate_features(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        bev_map: Optional[torch.Tensor] = None,
        extract_after_block: int = 15,
    ) -> torch.Tensor:
        """
        Run forward pass and extract intermediate features after a specific block.
        Used to provide visual features for cross-player attention.

        Args:
            extract_after_block: Block index after which to extract features.
                Default 15 (middle of 32 blocks) — captures mid-level features.
        Returns:
            features: [B, L, C] intermediate features.
        """
        self._extract_block = extract_after_block
        self._extracted_features = None

        # Temporarily hook the target block
        target_block = self.base_model.blocks[extract_after_block]
        orig_forward = target_block.forward

        def capture_forward(*args, **kwargs):
            result = orig_forward(*args, **kwargs)
            self._extracted_features = result.detach()
            return result

        target_block.forward = capture_forward

        try:
            _ = self.forward(x, t, context, seq_len, y, dit_cond_dict, bev_map)
        finally:
            target_block.forward = orig_forward

        return self._extracted_features


# ---------------------------------------------------------------------------
# Utility: count parameters
# ---------------------------------------------------------------------------

def count_stage2_params(wrapper: Stage2ModelWrapper) -> Dict[str, int]:
    """Count parameters by component."""
    bev_enc = sum(p.numel() for p in wrapper.bev_encoder.parameters())
    bev_attn = sum(
        sum(p.numel() for p in a.bev_cross_attn.parameters())
        for a in wrapper.block_adapters
    )
    xp_attn = sum(
        sum(p.numel() for p in a.cross_player_attn.parameters())
        for a in wrapper.block_adapters if a.has_cross_player
    )
    gates = sum(
        a.gate_bev.numel() + (a.gate_cross_player.numel() if a.has_cross_player else 0)
        for a in wrapper.block_adapters
    )
    total = bev_enc + bev_attn + xp_attn + gates
    return {
        "bev_encoder": bev_enc,
        "bev_cross_attn": bev_attn,
        "cross_player_attn": xp_attn,
        "gates": gates,
        "total": total,
    }
