"""
Stage 2 Training Script: Multi-View CSGO Video Generation

Phase 2a — BEV-only training:
  - Freezes all Stage 1 parameters (both high_noise_model and low_noise_model)
  - Trains: BEV encoder + BEV cross-attention (with zero-init gates)
  - Data: single player + BEV condition
  - Goal: model learns to query BEV for spatial awareness

Phase 2b — Cross-player attention training:
  - Freezes Stage 1 + optionally freezes/low-lr BEV modules
  - Trains: cross-player attention modules (with zero-init gates)
  - Data: 2-3 visible players from same episode
  - Goal: model learns to extract visual info from other players

Usage:
  # Phase 2a (BEV-only)
  accelerate launch --config_file accelerate_config_dual.yaml \\
      stage2/train_stage2.py \\
      --ckpt_dir /path/to/stage1_checkpoint/final \\
      --lingbot_code_dir /path/to/lingbot-world \\
      --dataset_dir /path/to/processed_csgo_v3 \\
      --stage2_dir /path/to/stage2_data \\
      --output_dir /path/to/output/stage2 \\
      --phase 2a

  # Phase 2b (add cross-player)
  accelerate launch --config_file accelerate_config_dual.yaml \\
      stage2/train_stage2.py \\
      --ckpt_dir /path/to/stage1_checkpoint/final \\
      --stage2_ckpt /path/to/stage2/phase2a_final \\
      --lingbot_code_dir /path/to/lingbot-world \\
      --dataset_dir /path/to/processed_csgo_v3 \\
      --stage2_dir /path/to/stage2_data \\
      --output_dir /path/to/output/stage2b \\
      --phase 2b
"""

import argparse
import logging
import os
import sys
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

# Stage 2 modules (relative import when run as module, or add to path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2.stage2_modules import Stage2ModelWrapper, count_stage2_params
from stage2.stage2_dataset import Stage2Dataset, stage2_collate_fn


# ============================================================
# Stage 2 Trainer
# ============================================================

class Stage2Trainer:
    """
    Extends Stage 1 training with BEV conditioning and cross-player attention.
    Reuses Stage 1's dual-model MoE strategy and flow matching loss.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        self.boundary = 0.947
        self.num_train_timesteps = 1000
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.shift = 10.0

        # Sigma schedule (same as Stage 1)
        sigmas_linear = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.shift * sigmas_linear / (1 + (self.shift - 1) * sigmas_linear)
        self.timesteps_schedule = self.sigmas * self.num_train_timesteps

        max_timestep = self.boundary * self.num_train_timesteps
        self.low_noise_indices = torch.where(self.timesteps_schedule < max_timestep)[0]
        self.high_noise_indices = torch.where(self.timesteps_schedule >= max_timestep)[0]

        # Training weights
        x = self.timesteps_schedule
        steps = self.num_train_timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        self.training_weights = y_shifted * (steps / y_shifted.sum())

        low_w = self.training_weights[self.low_noise_indices]
        self.low_noise_weights = low_w / low_w.mean()
        high_w = self.training_weights[self.high_noise_indices]
        self.high_noise_weights = high_w / high_w.mean()

    def load_models(self, device):
        """Load Stage 1 models and wrap with Stage 2 modules."""
        self.device = device
        ckpt_dir = self.args.ckpt_dir

        sys.path.insert(0, self.args.lingbot_code_dir)
        from wan.modules.model import WanModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel
        from wan.utils.cam_utils import (
            interpolate_camera_poses, compute_relative_poses,
            get_plucker_embeddings, get_Ks_transformed,
        )
        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }

        # Load base models
        logging.info("Loading Stage 1 low_noise_model...")
        low_base = WanModel.from_pretrained(
            ckpt_dir, subfolder="low_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )

        logging.info("Loading Stage 1 high_noise_model...")
        high_base = WanModel.from_pretrained(
            ckpt_dir, subfolder="high_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )

        # Wrap with Stage 2 modules
        enable_xp = (self.args.phase == "2b")
        logging.info(f"Wrapping with Stage2ModelWrapper (phase={self.args.phase})")

        self.low_noise_model = Stage2ModelWrapper(
            low_base,
            bev_channels=self.args.bev_channels,
            bev_size=self.args.bev_size,
            bev_token_grid=self.args.bev_token_grid,
            enable_cross_player=enable_xp,
        )

        self.high_noise_model = Stage2ModelWrapper(
            high_base,
            bev_channels=self.args.bev_channels,
            bev_size=self.args.bev_size,
            bev_token_grid=self.args.bev_token_grid,
            enable_cross_player=enable_xp,
        )

        # Freeze base models
        self.low_noise_model.freeze_base_model()
        self.high_noise_model.freeze_base_model()

        # Load Phase 2a checkpoint for Phase 2b
        if self.args.phase == "2b" and self.args.stage2_ckpt:
            self._load_stage2_checkpoint(self.args.stage2_ckpt)

        # Log parameter counts
        for name, model in [("low_noise", self.low_noise_model),
                             ("high_noise", self.high_noise_model)]:
            counts = count_stage2_params(model)
            logging.info(f"{name} Stage 2 params: {counts}")

        # Load VAE and T5
        logging.info("Loading VAE...")
        vae_path = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            base_dir = self.args.base_model_dir or ckpt_dir
            vae_path = os.path.join(base_dir, "Wan2.1_VAE.pth")
        self.vae = Wan2_1_VAE(
            vae_pth=vae_path,
            device=self.device,
        )

        logging.info("Loading T5 text encoder...")
        # T5 path: check both Stage 1 checkpoint and original model dir
        t5_path = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.exists(t5_path):
            # Might be in the base model dir (Stage 1 checkpoint doesn't always copy T5)
            base_dir = self.args.base_model_dir or ckpt_dir
            t5_path = os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth")

        tokenizer_path = os.path.join(ckpt_dir, "google", "umt5-xxl")
        if not os.path.exists(tokenizer_path):
            base_dir = self.args.base_model_dir or ckpt_dir
            tokenizer_path = os.path.join(base_dir, "google", "umt5-xxl")

        self.t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=t5_path,
            tokenizer_path=tokenizer_path,
        )

        return self.low_noise_model, self.high_noise_model

    def load_single_model(self, device, model_type="low"):
        """Load a SINGLE Stage 1 model + wrap with Stage 2 modules.
        Avoids OOM by only loading one 14B model at a time."""
        self.device = device
        ckpt_dir = self.args.ckpt_dir

        sys.path.insert(0, self.args.lingbot_code_dir)
        from wan.modules.model import WanModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel
        from wan.utils.cam_utils import (
            interpolate_camera_poses, compute_relative_poses,
            get_plucker_embeddings, get_Ks_transformed,
        )
        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }

        subfolder = "low_noise_model" if model_type == "low" else "high_noise_model"
        logging.info(f"Loading Stage 1 {subfolder}...")
        base_model = WanModel.from_pretrained(
            ckpt_dir, subfolder=subfolder,
            torch_dtype=torch.bfloat16, control_type="act",
        )

        enable_xp = (self.args.phase == "2b")
        wrapper = Stage2ModelWrapper(
            base_model,
            bev_channels=self.args.bev_channels,
            bev_size=self.args.bev_size,
            bev_token_grid=self.args.bev_token_grid,
            enable_cross_player=enable_xp,
        )
        wrapper.freeze_base_model()

        counts = count_stage2_params(wrapper)
        logging.info(f"{subfolder} Stage 2 params: {counts}")

        # Load Phase 2a checkpoint for Phase 2b
        if self.args.phase == "2b" and self.args.stage2_ckpt:
            path = os.path.join(self.args.stage2_ckpt, subfolder, "stage2_modules.pth")
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                missing, unexpected = wrapper.load_state_dict(state, strict=False)
                logging.info(f"Loaded Phase 2a checkpoint: {len(state)} params")

        # Load VAE
        logging.info("Loading VAE...")
        vae_path = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            base_dir = self.args.base_model_dir or ckpt_dir
            vae_path = os.path.join(base_dir, "Wan2.1_VAE.pth")
        self.vae = Wan2_1_VAE(vae_pth=vae_path, device=self.device)

        # Load T5
        logging.info("Loading T5 text encoder...")
        t5_path = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.exists(t5_path):
            base_dir = self.args.base_model_dir or ckpt_dir
            t5_path = os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_path = os.path.join(ckpt_dir, "google", "umt5-xxl")
        if not os.path.exists(tokenizer_path):
            base_dir = self.args.base_model_dir or ckpt_dir
            tokenizer_path = os.path.join(base_dir, "google", "umt5-xxl")
        self.t5 = T5EncoderModel(
            text_len=512, dtype=torch.bfloat16, device=self.device,
            checkpoint_path=t5_path, tokenizer_path=tokenizer_path,
        )

        return wrapper

    def _load_stage2_checkpoint(self, ckpt_path):
        """Load pre-trained Stage 2 modules (Phase 2a → 2b)."""
        logging.info(f"Loading Stage 2 checkpoint from {ckpt_path}")

        for name, model in [("low_noise_model", self.low_noise_model),
                             ("high_noise_model", self.high_noise_model)]:
            path = os.path.join(ckpt_path, name, "stage2_modules.pth")
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                # Load BEV encoder and block adapters
                missing, unexpected = model.load_state_dict(state, strict=False)
                logging.info(f"Loaded {name} Stage 2 modules: "
                             f"{len(state)} params, {len(missing)} missing, "
                             f"{len(unexpected)} unexpected")
            else:
                logging.warning(f"Stage 2 checkpoint not found: {path}")

    @torch.no_grad()
    def encode_video(self, video_tensor):
        latent = self.vae.encode([video_tensor.to(self.device)])[0]
        torch.cuda.empty_cache()
        return latent

    @torch.no_grad()
    def encode_text(self, prompt):
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()
        return [t.to(self.device) for t in context]

    def prepare_y(self, video_tensor, latent):
        """Same as Stage 1: mask + first frame VAE encoding."""
        lat_f, lat_h, lat_w = latent.shape[1], latent.shape[2], latent.shape[3]
        F_total = video_tensor.shape[1]
        h, w = video_tensor.shape[2], video_tensor.shape[3]

        first_frame = video_tensor[:, 0:1, :, :]
        zeros = torch.zeros(3, F_total - 1, h, w, device=video_tensor.device)
        vae_input = torch.concat([first_frame, zeros], dim=1)
        y_latent = self.vae.encode([vae_input.to(self.device)])[0]

        msk = torch.ones(1, F_total, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        y = torch.concat([msk, y_latent])
        return y

    def prepare_control_signal(self, poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w):
        """Same as Stage 1: rays_d(3) + wasd(4) = 7ch control signal."""
        interpolate_camera_poses = self.cam_utils["interpolate_camera_poses"]
        compute_relative_poses = self.cam_utils["compute_relative_poses"]
        get_plucker_embeddings = self.cam_utils["get_plucker_embeddings"]
        get_Ks_transformed = self.cam_utils["get_Ks_transformed"]

        num_frames = poses.shape[0]

        Ks = get_Ks_transformed(
            intrinsics,
            height_org=480, width_org=832,
            height_resize=h, width_resize=w,
            height_final=h, width_final=w,
        )
        Ks_single = Ks[0]

        len_c2ws = num_frames
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=poses[:, :3, :3].cpu().numpy(),
            src_trans_vec=poses[:, :3, 3].cpu().numpy(),
            tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)

        Ks_repeated = Ks_single.repeat(len(c2ws_infer), 1)
        c2ws_infer = c2ws_infer.to(self.device)
        Ks_repeated = Ks_repeated.to(self.device)

        wasd = actions[::4].to(self.device)
        if len(wasd) > len(c2ws_infer):
            wasd = wasd[:len(c2ws_infer)]
        elif len(wasd) < len(c2ws_infer):
            pad = wasd[-1:].repeat(len(c2ws_infer) - len(wasd), 1)
            wasd = torch.cat([wasd, pad], dim=0)

        c2ws_plucker_emb = get_plucker_embeddings(
            c2ws_infer, Ks_repeated, h, w, only_rays_d=True
        )

        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h), c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w,
        ).to(torch.bfloat16)

        wasd_tensor = wasd[:, None, None, :].repeat(1, h, w, 1)
        wasd_tensor = rearrange(
            wasd_tensor,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h), c2=int(w // lat_w),
        )
        wasd_tensor = wasd_tensor[None, ...]
        wasd_tensor = rearrange(
            wasd_tensor, 'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w,
        ).to(torch.bfloat16)

        c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_tensor], dim=1)

        dit_cond_dict = {
            "c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0),
        }
        return dit_cond_dict

    def training_step(self, model, batch, model_type="low"):
        """
        Training step with BEV conditioning (and optional cross-player features).

        Extends Stage 1 training_step by adding bev_map to the forward pass.
        """
        video = batch["video"].to(self.device)
        prompt = batch["prompt"]
        poses = batch["poses"]
        actions = batch["actions"]
        intrinsics = batch["intrinsics"]
        bev_map = batch["bev_map"].to(self.device)

        h, w = video.shape[2], video.shape[3]

        with torch.no_grad():
            video_latent = self.encode_video(video)

        lat_f = video_latent.shape[1]
        lat_h = video_latent.shape[2]
        lat_w = video_latent.shape[3]
        seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        with torch.no_grad():
            context = self.encode_text(prompt)
            y = self.prepare_y(video, video_latent)

        with torch.no_grad():
            dit_cond_dict = self.prepare_control_signal(
                poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
            )

        # Sample timestep (moved before Phase 2b so `t` is available for context feature extraction)
        if model_type == "high":
            indices = self.high_noise_indices
            local_idx = torch.randint(len(indices), (1,)).item()
            idx = indices[local_idx].item()
            training_weight = self.high_noise_weights[local_idx].item()
        else:
            indices = self.low_noise_indices
            local_idx = torch.randint(len(indices), (1,)).item()
            idx = indices[local_idx].item()
            training_weight = self.low_noise_weights[local_idx].item()

        sigma = self.sigmas[idx].item()
        t = self.timesteps_schedule[idx].to(self.device).unsqueeze(0)

        # --- Phase 2b: get cross-player features (after t is defined) ---
        visible_player_features = None
        if (self.args.phase == "2b"
                and "context_players" in batch
                and len(batch["context_players"]["videos"]) > 0):
            visible_player_features = self._extract_context_features(
                model, batch["context_players"], h, w, lat_f, lat_h, lat_w, seq_len, t
            )

        # Flow Matching: add noise
        noise = torch.randn_like(video_latent)
        noisy_latent = (1.0 - sigma) * video_latent + sigma * noise
        target = noise - video_latent

        # Forward through Stage 2 wrapper
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(
                x=[noisy_latent],
                t=t,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond_dict,
                bev_map=bev_map.unsqueeze(0),  # [1, C_bev, H, W]
                visible_player_features=visible_player_features,
            )[0]

        # Loss (exclude first temporal frame)
        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        loss = F.mse_loss(pred_rest.float(), target_rest.float())
        loss = loss * training_weight

        # Gate regularization: penalize large cross-player gates
        # to prevent over-reliance on context features (train-test gap mitigation)
        if self.args.phase == "2b" and self.args.gate_reg_weight > 0:
            gate_l2 = 0.0
            for adapter in model.block_adapters:
                if hasattr(adapter, 'gate_cross_player'):
                    gate_l2 = gate_l2 + adapter.gate_cross_player.pow(2).mean()
            loss = loss + self.args.gate_reg_weight * gate_l2

        return loss

    @torch.no_grad()
    def _extract_context_features(
        self, model, context_players, h, w, lat_f, lat_h, lat_w, seq_len,
        current_t=None,
    ):
        """
        Extract intermediate visual features from context players
        for cross-player attention.

        Adds Gaussian noise to extracted features (controlled by context_noise_std)
        to bridge the train-test gap: during training features come from GT video,
        during inference they come from noisy/generated video.

        Args:
            current_t: Current denoising timestep tensor.

        Returns concatenated features: [1, L_total, C]
        """
        all_features = []

        # Cache dummy text encoding to avoid repeated T5 GPU transfers
        if not hasattr(self, '_dummy_text_context'):
            self._dummy_text_context = self.encode_text(
                "A first-person view of a CS:GO match."
            )

        # Use current timestep if provided, otherwise fallback to mid
        if current_t is None:
            t_feat = torch.tensor([500.0], device=self.device)
        else:
            t_feat = current_t

        for i in range(len(context_players["videos"])):
            video_i = context_players["videos"][i].to(self.device)
            poses_i = context_players["poses"][i]
            actions_i = context_players["actions"][i]
            intrinsics_i = context_players["intrinsics"][i]

            # Encode video
            latent_i = self.encode_video(video_i)

            # Prepare inputs
            y_i = self.prepare_y(video_i, latent_i)
            dit_cond_i = self.prepare_control_signal(
                poses_i, actions_i, intrinsics_i, h, w, lat_f, lat_h, lat_w
            )

            # Extract mid-level features
            features_i = model.get_intermediate_features(
                x=[latent_i],
                t=t_feat,
                context=self._dummy_text_context,
                seq_len=seq_len,
                y=[y_i],
                dit_cond_dict=dit_cond_i,
                extract_after_block=15,
            )
            all_features.append(features_i)

        if all_features:
            result = torch.cat(all_features, dim=1)  # [1, L_total, C]

            # Add noise to context features to simulate inference conditions
            # where context comes from generated (noisy) video, not GT
            noise_std = getattr(self.args, 'context_noise_std', 0.1)
            if noise_std > 0:
                noise = torch.randn_like(result) * noise_std
                result = result + noise

            return result
        return None


# ============================================================
# Gradient Checkpointing for Stage 2
# ============================================================

def apply_gradient_checkpointing_stage2(wrapper, model_name="model"):
    """
    Apply gradient checkpointing to the Stage 2 adapted blocks.
    Since we patched the forward method, we need to wrap the patched version.
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    patched = 0
    for block in wrapper.base_model.blocks:
        if not any(p.requires_grad for p in block.parameters()):
            # Check adapter parameters too
            pass
        orig_forward = block.forward
        def _make_ckpt_fn(fn):
            @wraps(fn)
            def _ckpt_forward(x, e, seq_lens, grid_sizes, freqs, context,
                              context_lens, dit_cond_dict=None):
                return torch_checkpoint(
                    fn, x, e, seq_lens, grid_sizes, freqs, context,
                    context_lens, dit_cond_dict, use_reentrant=True
                )
            return _ckpt_forward
        block.forward = _make_ckpt_fn(orig_forward)
        patched += 1

    logging.info(f"Gradient checkpointing ({model_name}): patched {patched} blocks")


# ============================================================
# Checkpoint save/load
# ============================================================

def save_stage2_checkpoint(accelerator, low_wrapper, high_wrapper, args, tag):
    """Save only Stage 2 trainable parameters (dual-model version)."""
    import deepspeed

    save_dir = os.path.join(args.output_dir, tag)

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    for wrapper, name in [(low_wrapper, "low_noise_model"),
                           (high_wrapper, "high_noise_model")]:
        model_dir = os.path.join(save_dir, name)
        if accelerator.is_main_process:
            os.makedirs(model_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(wrapper)
        trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]

        with deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0):
            if accelerator.is_main_process:
                stage2_state = {}
                for n, p in unwrapped.named_parameters():
                    if p.requires_grad:
                        stage2_state[n] = p.data.cpu().clone()

                save_path = os.path.join(model_dir, "stage2_modules.pth")
                torch.save(stage2_state, save_path)
                size_mb = os.path.getsize(save_path) / (1024 * 1024)
                logging.info(f"Saved {name} Stage 2 modules "
                             f"({len(stage2_state)} params, {size_mb:.1f} MB) -> {save_path}")

    accelerator.wait_for_everyone()


def save_stage2_single_checkpoint(accelerator, wrapper, args, subfolder, tag):
    """Save Stage 2 trainable parameters for a single model.

    With DeepSpeed ZeRO-3, parameters are sharded across GPUs. Must use
    GatheredParameters to collect full tensors before saving on rank 0.
    """
    import deepspeed

    save_dir = os.path.join(args.output_dir, tag)
    model_dir = os.path.join(save_dir, subfolder)

    if accelerator.is_main_process:
        os.makedirs(model_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(wrapper)

    # Collect all trainable parameter objects for GatheredParameters
    trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]

    # Gather full parameters from all ZeRO-3 shards onto rank 0
    with deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0):
        if accelerator.is_main_process:
            stage2_state = {}
            for n, p in unwrapped.named_parameters():
                if p.requires_grad:
                    stage2_state[n] = p.data.cpu().clone()

            save_path = os.path.join(model_dir, "stage2_modules.pth")
            torch.save(stage2_state, save_path)
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            logging.info(f"Saved {subfolder} Stage 2 modules "
                         f"({len(stage2_state)} params, {size_mb:.1f} MB) -> {save_path}")

    accelerator.wait_for_everyone()


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Training: Multi-View CSGO Generation")

    # Model paths
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Stage 1 checkpoint directory")
    parser.add_argument("--stage2_ckpt", type=str, default="",
                        help="Stage 2 checkpoint to resume from (for Phase 2b)")
    parser.add_argument("--base_model_dir", type=str, default="",
                        help="Original model dir (for T5/VAE if not in ckpt_dir)")
    parser.add_argument("--lingbot_code_dir", type=str,
                        default="/home/nvme02/lingbot-world/code/lingbot-world")

    # Data paths
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Preprocessed Stage 1 data directory")
    parser.add_argument("--stage2_dir", type=str, required=True,
                        help="Stage 2 preprocessed data (BEV, visibility)")
    parser.add_argument("--output_dir", type=str, required=True)

    # Training config
    parser.add_argument("--model_type", type=str, required=True, choices=["low", "high"],
                        help="Which model to train: 'low' (t<947) or 'high' (t>=947)")
    parser.add_argument("--phase", type=str, choices=["2a", "2b"], default="2a")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Higher than Stage 1 since fewer parameters")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Data config
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--num_context_players", type=int, default=2,
                        help="Max visible players for Phase 2b")

    # BEV config
    parser.add_argument("--bev_channels", type=int, default=7,
                        help="Total BEV channels (4 static + 3 dynamic)")
    parser.add_argument("--bev_size", type=int, default=256,
                        help="BEV spatial resolution")
    parser.add_argument("--bev_token_grid", type=int, default=16,
                        help="BEV token grid size (tokens = grid^2)")

    # Cross-player robustness (Phase 2b)
    parser.add_argument("--context_noise_std", type=float, default=0.1,
                        help="Noise added to context features to bridge train-test gap")
    parser.add_argument("--gate_reg_weight", type=float, default=0.01,
                        help="L2 penalty on cross-player gates to prevent over-reliance")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    import accelerate
    from accelerate.utils import DataLoaderConfiguration
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Stage 2 Training — Phase {args.phase}")
        logging.info(f"Args: {args}")

    # Initialize trainer and load SINGLE model (like Stage 1)
    trainer = Stage2Trainer(args)
    model_type = args.model_type
    subfolder = "low_noise_model" if model_type == "low" else "high_noise_model"

    wrapper = trainer.load_single_model(accelerator.device, model_type)

    # Apply gradient checkpointing
    if args.gradient_checkpointing:
        apply_gradient_checkpointing_stage2(wrapper, subfolder)

    # Dataset
    dataset = Stage2Dataset(
        dataset_dir=args.dataset_dir,
        stage2_dir=args.stage2_dir,
        split="train",
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        bev_size=args.bev_size,
        phase=args.phase,
        num_context_players=args.num_context_players,
        repeat=args.dataset_repeat,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=stage2_collate_fn,
    )

    # Optimizer — only Stage 2 parameters
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
    logging.info(f"Trainable params ({subfolder}): {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    import math
    iters_per_epoch = math.ceil(len(dataset) / accelerator.num_processes)
    steps_per_epoch = iters_per_epoch // args.gradient_accumulation_steps
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-6
    )

    # Prepare with accelerator — single engine
    wrapper, optimizer, dataloader, scheduler = accelerator.prepare(
        wrapper, optimizer, dataloader, scheduler
    )

    # Pre-warm T5 cache
    if accelerator.is_main_process:
        logging.info("Pre-warming T5 cache...")
    _ = trainer.encode_text("First-person view of CS:GO competitive gameplay")
    trainer.t5.model.cpu()
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logging.info(f"Training config: {args.num_epochs} epochs, "
                     f"{len(dataloader)} iters/epoch, "
                     f"grad_accum={args.gradient_accumulation_steps}, "
                     f"optimizer_steps/epoch={steps_per_epoch}, "
                     f"total_optimizer_steps={total_steps}")

    # ============================================================
    # Training loop (single model, like Stage 1)
    # ============================================================
    global_step = 0
    diagnosed = False  # early diagnostic flag

    for epoch in range(args.num_epochs):
        wrapper.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader, disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch+1}/{args.num_epochs} [{subfolder}]"
        )

        for batch in progress:
            with accelerator.accumulate(wrapper):
                loss = trainer.training_step(
                    accelerator.unwrap_model(wrapper), batch, model_type
                )
                accelerator.backward(loss)

                # --- Early diagnostic: check gradients AFTER backward, BEFORE zero_grad ---
                if not diagnosed and global_step >= 2 and accelerator.sync_gradients and accelerator.is_main_process:
                    diagnosed = True
                    unwrapped = accelerator.unwrap_model(wrapper)
                    logging.info("=" * 60)
                    logging.info("EARLY DIAGNOSTIC (after backward, before zero_grad)")

                    # Check gate values (should start near 0, slowly increase)
                    import deepspeed
                    gate_params = [a.gate_bev for a in unwrapped.block_adapters]
                    with deepspeed.zero.GatheredParameters(gate_params, modifier_rank=0):
                        for i, adapter in enumerate(unwrapped.block_adapters):
                            gate_val = adapter.gate_bev.abs().mean().item()
                            logging.info(f"  block_adapter[{i}] gate_bev abs mean: {gate_val:.6f}")

                    # Check if any trainable param has non-zero gradient
                    # Note: with ZeRO-3, p.grad on rank 0 may only show local shard's grad
                    has_grad = 0
                    no_grad = 0
                    for n, p in unwrapped.named_parameters():
                        if p.requires_grad:
                            if p.grad is not None and p.grad.abs().sum() > 0:
                                has_grad += 1
                            else:
                                no_grad += 1
                    logging.info(f"  Trainable params with gradient: {has_grad}, without: {no_grad}")
                    if has_grad == 0:
                        logging.warning("  Note: with ZeRO-3, gradient check on rank 0 may show 0 "
                                        "due to gradient partitioning. Check gate values instead — "
                                        "if gates change from 0, training IS working.")

                    # Test checkpoint save (GatheredParameters sanity check)
                    test_params = [p for p in unwrapped.parameters() if p.requires_grad]
                    with deepspeed.zero.GatheredParameters(test_params, modifier_rank=0):
                        total_numel = sum(p.numel() for p in test_params)
                        logging.info(f"  GatheredParameters total elements: {total_numel:,}")
                        expected_mb = total_numel * 4 / (1024 * 1024)
                        logging.info(f"  Expected checkpoint size: ~{expected_mb:.0f} MB")
                        if total_numel < 1000:
                            logging.error("  WARNING: GatheredParameters returned very few elements! "
                                          "Checkpoint save will be broken.")

                    logging.info("=" * 60)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(wrapper.parameters(), args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            if accelerator.sync_gradients:
                global_step += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                step=global_step,
            )

        avg_loss = epoch_loss / max(num_batches, 1)
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} [{subfolder}] - avg_loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_stage2_single_checkpoint(accelerator, wrapper, args, subfolder, f"epoch_{epoch+1}")

            # Verify checkpoint size immediately after save
            if accelerator.is_main_process:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}",
                                         subfolder, "stage2_modules.pth")
                if os.path.exists(ckpt_path):
                    ckpt_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                    logging.info(f"  Checkpoint size: {ckpt_mb:.1f} MB")
                    if ckpt_mb < 10:
                        logging.error(f"  WARNING: Checkpoint too small ({ckpt_mb:.1f} MB)! "
                                      f"Expected ~200-500 MB. ZeRO-3 gather may have failed.")

    save_stage2_single_checkpoint(accelerator, wrapper, args, subfolder, "final")
    if accelerator.is_main_process:
        logging.info(f"Stage 2 training complete! {subfolder}")


if __name__ == "__main__":
    main()
