"""
Stage 2 Multi-Player Inference Script.

Generates N geometrically-consistent first-person videos from:
  - N players' first frames + action sequences
  - Shared BEV map (static + dynamic)
  - Pre-computed or on-the-fly visibility matrix

Denoising loop:
  For each timestep t:
    1. Select model (high_noise if t >= boundary, else low_noise)
    2. Feature extraction pass: run all players through model (no_grad),
       extract intermediate features from block 15
    3. Denoising pass: run all players with cross-player features,
       apply CFG, UniPC solver step

Usage:
    python stage2/inference_stage2.py \\
        --base_model_dir /path/to/lingbot-world-base-act \\
        --stage1_ckpt /path/to/stage1/final \\
        --stage2_ckpt /path/to/stage2/final \\
        --lingbot_code_dir /path/to/lingbot-world \\
        --player_dirs player0_clip,player1_clip,player2_clip \\
        --bev_path /path/to/stage2_data \\
        --output_dir ./output_multiview \\
        --sampling_steps 70
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2.stage2_modules import Stage2ModelWrapper


# ============================================================
# Noise schedule and solver utilities
# ============================================================

def build_sigma_schedule(num_timesteps: int = 1000, shift: float = 10.0):
    """Build the shifted sigma schedule (same as training)."""
    sigmas_linear = torch.linspace(1.0, 0.0, num_timesteps + 1)[:-1]
    sigmas = shift * sigmas_linear / (1 + (shift - 1) * sigmas_linear)
    timesteps = sigmas * num_timesteps
    return sigmas, timesteps


def get_sampling_timesteps(num_steps: int, num_timesteps: int = 1000, shift: float = 10.0):
    """
    Get timestep schedule for inference (evenly spaced in sigma space).
    Returns list of timesteps in descending order.
    """
    sigmas, _ = build_sigma_schedule(num_timesteps, shift)

    # Evenly spaced indices
    indices = torch.linspace(0, num_timesteps - 1, num_steps + 1).long()
    timestep_values = sigmas[indices] * num_timesteps

    # Return descending (high noise to low noise), skip the last (t=0)
    return timestep_values[:-1].flip(0)


# ============================================================
# Simple flow-matching Euler solver (fallback if UniPC not available)
# ============================================================

def euler_step(latent, noise_pred, sigma_curr, sigma_next):
    """
    Single Euler step for flow matching.
    velocity = noise - clean  =>  clean = sample - sigma * velocity
    """
    x0_pred = latent - sigma_curr * noise_pred
    if sigma_next > 0:
        return sigma_next * noise_pred + (1.0 - sigma_next) * x0_pred
    return x0_pred


# ============================================================
# Multi-player inference engine
# ============================================================

class MultiPlayerInference:
    """
    Orchestrates multi-player video generation with Stage 2 model.

    For each denoising step:
      1. Select model (dual-model MoE by timestep)
      2. Extract intermediate features from all players (for cross-player attn)
      3. Run denoising with BEV + cross-player features
      4. Apply classifier-free guidance
      5. Solver step (Euler or UniPC)
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        self.boundary = 0.947
        self.num_train_timesteps = 1000
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.shift = args.sample_shift

        # Sigma schedule
        self.sigmas, self.timestep_schedule = build_sigma_schedule(
            self.num_train_timesteps, self.shift
        )

    def load_models(self):
        """Load and wrap both models with Stage 2 modules."""
        sys.path.insert(0, self.args.lingbot_code_dir)
        from wan.modules.model import WanModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel
        from wan.utils.cam_utils import (
            interpolate_camera_poses, compute_relative_poses,
            get_plucker_embeddings, get_Ks_transformed,
        )
        self.cam_utils = {
            "interpolate": interpolate_camera_poses,
            "relative": compute_relative_poses,
            "plucker": get_plucker_embeddings,
            "Ks": get_Ks_transformed,
        }

        base_dir = self.args.base_model_dir
        stage1_dir = self.args.stage1_ckpt
        stage2_dir = self.args.stage2_ckpt

        # Determine where to load base models from
        # Priority: stage1_ckpt > base_model_dir
        low_model_dir = stage1_dir if stage1_dir else base_dir
        high_model_dir = stage1_dir if stage1_dir else base_dir

        logging.info(f"Loading low_noise_model from {low_model_dir}")
        low_base = WanModel.from_pretrained(
            low_model_dir, subfolder="low_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )
        logging.info(f"Loading high_noise_model from {high_model_dir}")
        high_base = WanModel.from_pretrained(
            high_model_dir, subfolder="high_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )

        # Wrap with Stage 2 modules
        self.low_model = Stage2ModelWrapper(
            low_base,
            bev_channels=self.args.bev_channels,
            bev_size=self.args.bev_size,
            bev_token_grid=self.args.bev_token_grid,
            enable_cross_player=self.args.enable_cross_player,
        )
        self.high_model = Stage2ModelWrapper(
            high_base,
            bev_channels=self.args.bev_channels,
            bev_size=self.args.bev_size,
            bev_token_grid=self.args.bev_token_grid,
            enable_cross_player=self.args.enable_cross_player,
        )

        # Load Stage 2 trained weights
        if stage2_dir:
            self._load_stage2_weights(stage2_dir)

        # Freeze everything (inference mode)
        self.low_model.eval()
        self.high_model.eval()

        # Move to device
        self.low_model.to(self.device)
        self.high_model.to(self.device)

        # Load VAE
        vae_path = os.path.join(base_dir, "Wan2.1_VAE.pth")
        logging.info(f"Loading VAE from {vae_path}")
        self.vae = Wan2_1_VAE(vae_pth=vae_path, device=self.device)

        # Load T5
        t5_path = os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_path = os.path.join(base_dir, "google", "umt5-xxl")
        logging.info(f"Loading T5 from {t5_path}")
        self.t5 = T5EncoderModel(
            text_len=512, dtype=torch.bfloat16, device=self.device,
            checkpoint_path=t5_path, tokenizer_path=tokenizer_path,
        )

        logging.info("All models loaded.")

    def _load_stage2_weights(self, stage2_dir):
        """Load Stage 2 module weights into both wrappers."""
        for name, model in [("low_noise_model", self.low_model),
                             ("high_noise_model", self.high_model)]:
            path = os.path.join(stage2_dir, name, "stage2_modules.pth")
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                missing, unexpected = model.load_state_dict(state, strict=False)
                logging.info(f"Loaded {name} Stage 2: {len(state)} params, "
                             f"{len(missing)} missing, {len(unexpected)} unexpected")
            else:
                logging.warning(f"Stage 2 weights not found: {path}")

    def select_model(self, t_value: float) -> Stage2ModelWrapper:
        """Select high_noise or low_noise model based on timestep."""
        boundary = self.boundary * self.num_train_timesteps
        if t_value >= boundary:
            return self.high_model
        return self.low_model

    # ============================================================
    # Data preparation (per-player)
    # ============================================================

    def load_player_data(self, clip_dir: str) -> dict:
        """
        Load a player's clip data: first frame image, poses, actions, intrinsics.
        """
        # First frame image
        image_path = os.path.join(clip_dir, "image.jpg")
        if not os.path.exists(image_path):
            # Extract first frame from video
            import cv2
            video_path = os.path.join(clip_dir, "video.mp4")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
            else:
                raise FileNotFoundError(f"Cannot read video: {video_path}")
        else:
            img = Image.open(image_path).convert("RGB")

        # Resize to target resolution
        img = img.resize((self.args.width, self.args.height), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

        # Load numpy data
        poses = torch.from_numpy(np.load(os.path.join(clip_dir, "poses.npy"))).float()
        actions = torch.from_numpy(np.load(os.path.join(clip_dir, "action.npy"))).float()
        intrinsics = torch.from_numpy(np.load(os.path.join(clip_dir, "intrinsics.npy"))).float()

        return {
            "image": img_tensor,       # [3, H, W]
            "poses": poses,            # [F, 4, 4]
            "actions": actions,        # [F, 7]
            "intrinsics": intrinsics,  # [F, 3, 3]
        }

    @torch.no_grad()
    def encode_image_condition(self, img_tensor: torch.Tensor, num_frames: int):
        """
        Encode first frame into VAE latent + mask (same as prepare_y in training).

        Args:
            img_tensor: [3, H, W]
            num_frames: Total video frames (e.g., 81)

        Returns:
            y: [20, lat_f, lat_h, lat_w]  (4ch mask + 16ch VAE latent)
            latent_shape: (lat_f, lat_h, lat_w)
        """
        h, w = img_tensor.shape[1], img_tensor.shape[2]

        # Build VAE input: first frame + zeros
        first_frame = img_tensor.unsqueeze(1)  # [3, 1, H, W]
        zeros = torch.zeros(3, num_frames - 1, h, w, device=img_tensor.device)
        vae_input = torch.cat([first_frame, zeros], dim=1)  # [3, F, H, W]

        y_latent = self.vae.encode([vae_input.to(self.device)])[0]  # [16, lat_f, lat_h, lat_w]
        lat_f, lat_h, lat_w = y_latent.shape[1], y_latent.shape[2], y_latent.shape[3]

        # Build mask: frame 0 = 1 (known), rest = 0
        msk = torch.ones(1, num_frames, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, lat_f, lat_h, lat_w]

        y = torch.cat([msk, y_latent])  # [20, lat_f, lat_h, lat_w]
        return y, (lat_f, lat_h, lat_w)

    def prepare_control_signal(self, poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w):
        """Build camera + action control signal (same as training)."""
        num_frames = poses.shape[0]

        Ks = self.cam_utils["Ks"](
            intrinsics, height_org=480, width_org=832,
            height_resize=h, width_resize=w,
            height_final=h, width_final=w,
        )
        Ks_single = Ks[0]

        c2ws = self.cam_utils["interpolate"](
            src_indices=np.linspace(0, num_frames - 1, num_frames),
            src_rot_mat=poses[:, :3, :3].cpu().numpy(),
            src_trans_vec=poses[:, :3, 3].cpu().numpy(),
            tgt_indices=np.linspace(0, num_frames - 1, int((num_frames - 1) // 4) + 1),
        )
        c2ws = self.cam_utils["relative"](c2ws, framewise=True)

        Ks_rep = Ks_single.repeat(len(c2ws), 1).to(self.device)
        c2ws = c2ws.to(self.device)

        wasd = actions[::4].to(self.device)
        if len(wasd) > len(c2ws):
            wasd = wasd[:len(c2ws)]
        elif len(wasd) < len(c2ws):
            pad = wasd[-1:].repeat(len(c2ws) - len(wasd), 1)
            wasd = torch.cat([wasd, pad], dim=0)

        rays = self.cam_utils["plucker"](c2ws, Ks_rep, h, w, only_rays_d=True)

        rays = rearrange(
            rays, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h), c2=int(w // lat_w),
        )[None]
        rays = rearrange(rays, 'b (f h w) c -> b c f h w',
                         f=lat_f, h=lat_h, w=lat_w).to(torch.bfloat16)

        wasd_t = wasd[:, None, None, :].repeat(1, h, w, 1)
        wasd_t = rearrange(
            wasd_t, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h), c2=int(w // lat_w),
        )[None]
        wasd_t = rearrange(wasd_t, 'b (f h w) c -> b c f h w',
                           f=lat_f, h=lat_h, w=lat_w).to(torch.bfloat16)

        ctrl = torch.cat([rays, wasd_t], dim=1)
        return {"c2ws_plucker_emb": ctrl.chunk(1, dim=0)}

    @torch.no_grad()
    def encode_text(self, prompt: str):
        """Encode text prompt."""
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()
        return [t.to(self.device) for t in context]

    # ============================================================
    # Main multi-player generation
    # ============================================================

    @torch.no_grad()
    def generate(
        self,
        player_dirs: List[str],
        bev_maps: List[torch.Tensor],
        visibility_matrix: Optional[np.ndarray] = None,
        prompt: str = "First-person view of CS:GO competitive gameplay",
    ) -> List[torch.Tensor]:
        """
        Generate videos for all players simultaneously.

        Args:
            player_dirs: List of clip directories, one per player.
            bev_maps: List of N tensors [C_bev, H_bev, W_bev], one per player.
            visibility_matrix: [N, N] average visibility (or None for no cross-player).
            prompt: Text prompt (shared across all players).

        Returns:
            videos: List of [3, F, H, W] tensors, one per player.
        """
        N = len(player_dirs)
        num_frames = self.args.frame_num
        h, w = self.args.height, self.args.width
        guide_scale = self.args.guide_scale
        sampling_steps = self.args.sampling_steps

        logging.info(f"Generating {N} player videos, {num_frames} frames, {sampling_steps} steps")

        # ---- Step 1: Load and prepare all players' data ----
        logging.info("Loading player data...")
        players = []
        for i, pdir in enumerate(player_dirs):
            data = self.load_player_data(pdir)
            players.append(data)

        # ---- Step 2: Encode text (shared) ----
        logging.info("Encoding text...")
        context_cond = self.encode_text(prompt)
        context_null = self.encode_text("")

        # ---- Step 3: Encode first frames + prepare control signals ----
        logging.info("Encoding player conditions...")
        player_states = []

        for i, p in enumerate(players):
            img = p["image"].to(self.device)
            y, (lat_f, lat_h, lat_w) = self.encode_image_condition(img, num_frames)
            seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

            dit_cond = self.prepare_control_signal(
                p["poses"], p["actions"], p["intrinsics"],
                h, w, lat_f, lat_h, lat_w,
            )

            # Initialize noise
            noise = torch.randn(16, lat_f, lat_h, lat_w, device=self.device,
                                generator=torch.Generator(device=self.device).manual_seed(42 + i))

            player_states.append({
                "latent": noise,
                "y": y,
                "dit_cond": dit_cond,
                "seq_len": seq_len,
                "lat_shape": (lat_f, lat_h, lat_w),
            })

        # Per-player BEV maps
        bev_tensors = [bm.unsqueeze(0).to(self.device) for bm in bev_maps]  # List of [1, C, H, W]

        # ---- Step 4: Build visibility lookup ----
        if visibility_matrix is not None and self.args.enable_cross_player:
            vis = visibility_matrix  # [N, N]
        else:
            vis = None

        # ---- Step 5: Denoising loop ----
        timesteps = get_sampling_timesteps(sampling_steps, self.num_train_timesteps, self.shift)
        sigmas, _ = build_sigma_schedule(self.num_train_timesteps, self.shift)

        logging.info(f"Starting denoising loop ({len(timesteps)} steps)...")
        start_time = time.time()

        for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising")):
            t_val = t.item()
            model = self.select_model(t_val)

            # Current and next sigma
            sigma_idx = (self.timestep_schedule - t_val).abs().argmin().item()
            sigma_curr = sigmas[sigma_idx].item()
            if step_idx + 1 < len(timesteps):
                next_t = timesteps[step_idx + 1].item()
                next_idx = (self.timestep_schedule - next_t).abs().argmin().item()
                sigma_next = sigmas[next_idx].item()
            else:
                sigma_next = 0.0

            t_tensor = torch.tensor([t_val], device=self.device)

            # ---- Phase A: Extract features from all players (for cross-player attn) ----
            player_features = {}
            if vis is not None and self.args.enable_cross_player:
                for i in range(N):
                    ps = player_states[i]
                    features_i = model.get_intermediate_features(
                        x=[ps["latent"]],
                        t=t_tensor,
                        context=context_cond,
                        seq_len=ps["seq_len"],
                        y=[ps["y"]],
                        dit_cond_dict=ps["dit_cond"],
                        bev_map=bev_tensors[i],
                        extract_after_block=15,
                    )
                    player_features[i] = features_i

            # ---- Phase B: Denoising pass for each player ----
            for i in range(N):
                ps = player_states[i]

                # Collect visible player features
                visible_feats = None
                if vis is not None and self.args.enable_cross_player:
                    vis_player_feats = []
                    for j in range(N):
                        if j != i and vis[i, j] > 0.1:
                            vis_player_feats.append(player_features[j])
                    if vis_player_feats:
                        visible_feats = torch.cat(vis_player_feats, dim=1)

                # Conditional forward
                noise_pred_cond = model(
                    x=[ps["latent"]],
                    t=t_tensor,
                    context=context_cond,
                    seq_len=ps["seq_len"],
                    y=[ps["y"]],
                    dit_cond_dict=ps["dit_cond"],
                    bev_map=bev_tensors[i],
                    visible_player_features=visible_feats,
                )[0]

                # Unconditional forward (no text guidance, but keep control signals)
                noise_pred_uncond = model(
                    x=[ps["latent"]],
                    t=t_tensor,
                    context=context_null,
                    seq_len=ps["seq_len"],
                    y=[ps["y"]],
                    dit_cond_dict=ps["dit_cond"],
                    bev_map=bev_tensors[i],
                    visible_player_features=visible_feats,
                )[0]

                # Classifier-free guidance
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                # Euler step
                ps["latent"] = euler_step(ps["latent"], noise_pred, sigma_curr, sigma_next)

        elapsed = time.time() - start_time
        logging.info(f"Denoising complete in {elapsed:.1f}s "
                     f"({elapsed / sampling_steps:.2f}s/step, "
                     f"{elapsed / sampling_steps / N:.2f}s/step/player)")

        # ---- Step 6: Decode latents to videos ----
        logging.info("Decoding videos...")
        videos = []
        for i in range(N):
            video = self.vae.decode([player_states[i]["latent"]])[0]  # [3, F, H, W]
            videos.append(video.cpu())
            torch.cuda.empty_cache()

        return videos


# ============================================================
# BEV + Visibility: build on-the-fly from poses (self-contained)
# ============================================================

def extract_positions_and_yaws_from_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract world positions and yaw angles from camera pose matrices.

    Args:
        poses: [F, 4, 4] camera-to-world matrices.

    Returns:
        positions: [F, 3] xyz world positions.
        yaws: [F] yaw angles in degrees.
    """
    # Position = translation column of c2w matrix
    positions = poses[:, :3, 3]  # [F, 3]

    # Yaw = atan2 of forward direction projected onto XY plane
    # Forward direction is the negative Z axis in camera space = -R[:, 2]
    forward = -poses[:, :3, 2]  # [F, 3]
    yaws = np.degrees(np.arctan2(forward[:, 1], forward[:, 0]))  # [F]

    return positions, yaws


def build_bev_from_poses(
    all_player_poses: List[np.ndarray],
    bev_size: int = 256,
    static_bev_path: Optional[str] = None,
    navmesh_path: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Build per-player BEV maps directly from player poses (no pre-processing needed).

    Each player gets their own BEV where dynamic channels (friendly/enemy/self)
    are correct from their perspective.

    Static BEV (4 channels): from navmesh > cached .npy > zeros.
    Dynamic BEV (3 channels): built per-player from positions.
    Total: 7 channels.

    Args:
        all_player_poses: List of [F, 4, 4] pose arrays, one per player.
        bev_size: BEV spatial resolution.
        static_bev_path: Optional path to pre-computed static_bev.npy.
        navmesh_path: Optional path to navmesh.json (preferred over static_bev_path).

    Returns:
        bev_maps: List of N tensors, each [7, bev_size, bev_size].
    """
    from stage2.bev_builder import (
        StaticBEVBuilder, DynamicBEVBuilder, estimate_bounds_from_positions,
        NUM_STATIC_CHANNELS, NUM_DYNAMIC_CHANNELS,
    )

    N = len(all_player_poses)

    # ---- Static BEV (4 channels) ----
    static_builder = StaticBEVBuilder(bev_size=bev_size)
    if navmesh_path and os.path.exists(navmesh_path):
        static_bev = static_builder.build_from_navmesh(navmesh_path)
        bounds = static_builder.map_bounds
    elif static_bev_path and os.path.exists(static_bev_path):
        static_bev, bounds = static_builder.load_cached(static_bev_path)
        logging.info(f"Loaded static BEV from {static_bev_path}")
    else:
        static_bev = np.zeros((NUM_STATIC_CHANNELS, bev_size, bev_size), dtype=np.float32)
        bounds = None
        logging.info("No static BEV or navmesh provided, using zeros (position-only mode)")

    # ---- Extract positions and yaws from all players ----
    all_positions = []
    all_yaws = []
    for poses in all_player_poses:
        pos, yaw = extract_positions_and_yaws_from_poses(poses)
        all_positions.append(pos)
        all_yaws.append(yaw)

    # ---- Estimate map bounds if not set by static BEV ----
    if bounds is None:
        all_pos_flat = np.concatenate(all_positions, axis=0)
        bounds = estimate_bounds_from_positions(all_pos_flat, padding_factor=1.3)
    logging.info(f"Map bounds: x=[{bounds['x_min']:.0f}, {bounds['x_max']:.0f}], "
                 f"y=[{bounds['y_min']:.0f}, {bounds['y_max']:.0f}]")

    # ---- Build per-player dynamic BEV (3 channels), averaged over frames ----
    builder = DynamicBEVBuilder(bev_size=bev_size)
    builder.set_map_bounds(bounds)

    num_frames = min(p.shape[0] for p in all_player_poses)
    subsample = max(1, num_frames // 20)
    sampled_frames = list(range(0, num_frames, subsample))

    bev_maps = []
    for player_idx in range(N):
        dynamic_accum = np.zeros((NUM_DYNAMIC_CHANNELS, bev_size, bev_size), dtype=np.float32)
        player_team = 0 if player_idx < N // 2 else 1

        for t in sampled_frames:
            frame_players = []
            for p_idx in range(N):
                frame_players.append({
                    "x": float(all_positions[p_idx][t, 0]),
                    "y": float(all_positions[p_idx][t, 1]),
                    "yaw": float(all_yaws[p_idx][t]),
                    "team_id": 0 if p_idx < N // 2 else 1,
                    "alive": True,
                })
            dynamic_accum += builder.build_frame(
                current_player_idx=player_idx,
                current_player_team=player_team,
                all_players=frame_players,
            )

        dynamic_bev = dynamic_accum / len(sampled_frames)
        bev_map = np.concatenate([static_bev, dynamic_bev], axis=0)  # [7, H, W]
        bev_maps.append(torch.from_numpy(bev_map).float())

    return bev_maps


def build_visibility_from_poses(
    all_player_poses: List[np.ndarray],
    fov_degrees: float = 90.0,
) -> np.ndarray:
    """
    Compute average visibility matrix from player poses (no pre-processing needed).

    Args:
        all_player_poses: List of [F, 4, 4] pose arrays, one per player.
        fov_degrees: Horizontal field of view in degrees.

    Returns:
        vis_matrix: [N, N] averaged visibility matrix.
    """
    from stage2.visibility import VisibilityComputer

    N = len(all_player_poses)
    computer = VisibilityComputer(fov_degrees=fov_degrees)

    # Extract positions and yaws
    all_positions = []
    all_yaws = []
    for poses in all_player_poses:
        pos, yaw = extract_positions_and_yaws_from_poses(poses)
        all_positions.append(pos)
        all_yaws.append(yaw)

    num_frames = min(p.shape[0] for p in all_player_poses)
    # Subsample for efficiency
    subsample = max(1, num_frames // 20)
    sampled_frames = list(range(0, num_frames, subsample))

    vis_accum = np.zeros((N, N), dtype=np.float32)

    for t in sampled_frames:
        players = []
        for p_idx in range(N):
            players.append({
                "x": float(all_positions[p_idx][t, 0]),
                "y": float(all_positions[p_idx][t, 1]),
                "z": float(all_positions[p_idx][t, 2]),
                "yaw": float(all_yaws[p_idx][t]),
                "alive": True,
            })
        vis_accum += computer.compute_frame(players)

    vis_matrix = vis_accum / len(sampled_frames)
    logging.info(f"Built visibility matrix from poses: "
                 f"avg visible pairs per player = {vis_matrix.sum(axis=1).mean():.1f}")
    return vis_matrix


def load_bev(stage2_data_dir: str, episode_id: str = None, player_stem: str = None):
    """
    Load pre-computed BEV (static + dynamic) from preprocessed files.
    Fallback path — prefer build_bev_from_poses() for self-contained inference.

    Returns:
        bev_map: [C_total, H, W] tensor
        vis_matrix: [N, N] numpy array or None
    """
    # Static BEV
    static_path = os.path.join(stage2_data_dir, "static_bev.npy")
    if os.path.exists(static_path):
        static_bev = np.load(static_path)
    else:
        logging.warning("Static BEV not found, using zeros")
        static_bev = np.zeros((4, 256, 256), dtype=np.float32)

    # Dynamic BEV (episode-specific)
    dynamic_bev = None
    if episode_id and player_stem:
        dyn_path = os.path.join(
            stage2_data_dir, "episodes", f"Ep_{episode_id}",
            f"dynamic_bev_{player_stem}.npy"
        )
        if os.path.exists(dyn_path):
            dyn = np.load(dyn_path)
            if dyn.ndim == 4:
                dynamic_bev = dyn.mean(axis=0).astype(np.float32)
            else:
                dynamic_bev = dyn.astype(np.float32)

    if dynamic_bev is None:
        dynamic_bev = np.zeros((3, 256, 256), dtype=np.float32)
    elif dynamic_bev.shape[0] > 3:
        # Legacy 6-channel dynamic BEV: take only first 3 (friendly/enemy/self)
        dynamic_bev = dynamic_bev[:3]

    bev_map = np.concatenate([static_bev, dynamic_bev], axis=0)

    # Visibility matrix
    vis_matrix = None
    if episode_id:
        vis_path = os.path.join(
            stage2_data_dir, "episodes", f"Ep_{episode_id}", "visibility.npy"
        )
        if os.path.exists(vis_path):
            vis = np.load(vis_path)
            vis_matrix = vis.mean(axis=0)

    return torch.from_numpy(bev_map).float(), vis_matrix


# ============================================================
# Video saving utility
# ============================================================

def save_video_mp4(video_tensor, save_path, fps=16):
    """
    Save [3, F, H, W] tensor (range [-1, 1]) as MP4 file.
    """
    import cv2

    video = video_tensor.clamp(-1, 1)
    video = ((video + 1) / 2 * 255).byte()  # [3, F, H, W] in [0, 255]

    F_total = video.shape[1]
    H, W = video.shape[2], video.shape[3]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for t in range(F_total):
        frame = video[:, t].permute(1, 2, 0).numpy()  # [H, W, 3] RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    logging.info(f"Saved video ({F_total} frames, {H}x{W}) -> {save_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Multi-Player Inference")

    # Model paths
    parser.add_argument("--base_model_dir", type=str, required=True,
                        help="Original lingbot-world-base-act model dir")
    parser.add_argument("--stage1_ckpt", type=str, default="",
                        help="Stage 1 fine-tuned checkpoint")
    parser.add_argument("--stage2_ckpt", type=str, default="",
                        help="Stage 2 trained checkpoint")
    parser.add_argument("--lingbot_code_dir", type=str,
                        default="/home/nvme02/lingbot-world/code/lingbot-world")

    # Input data
    parser.add_argument("--player_dirs", type=str, required=True,
                        help="Comma-separated paths to player clip directories")
    parser.add_argument("--bev_dir", type=str, default="",
                        help="Stage 2 preprocessed data dir (BEV + visibility). "
                             "If not set, BEV and visibility are built on-the-fly from poses.")
    parser.add_argument("--episode_id", type=str, default="",
                        help="Episode ID for loading dynamic BEV + visibility (only with --bev_dir)")
    parser.add_argument("--player_stems", type=str, default="",
                        help="Comma-separated player stems (for dynamic BEV, only with --bev_dir)")
    parser.add_argument("--static_bev_path", type=str, default="",
                        help="Optional path to static_bev.npy (map geometry). "
                             "Used in on-the-fly mode for the 4 static channels.")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output_multiview")
    parser.add_argument("--prompt", type=str,
                        default="First-person view of CS:GO competitive gameplay on de_dust2")

    # Generation config
    parser.add_argument("--sampling_steps", type=int, default=70)
    parser.add_argument("--sample_shift", type=float, default=10.0)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # BEV config
    parser.add_argument("--bev_channels", type=int, default=7,
                        help="Total BEV channels (4 static + 3 dynamic)")
    parser.add_argument("--bev_size", type=int, default=256)
    parser.add_argument("--bev_token_grid", type=int, default=16)
    parser.add_argument("--navmesh_path", type=str, default="",
                        help="Path to navmesh.json (preferred for static BEV)")
    parser.add_argument("--enable_cross_player", action="store_true",
                        help="Enable cross-player attention (Phase 2b)")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse player directories
    player_dirs = [d.strip() for d in args.player_dirs.split(",")]
    N = len(player_dirs)
    logging.info(f"Generating for {N} players")

    # ---- Load player poses for on-the-fly BEV/visibility ----
    all_player_poses = []
    for pdir in player_dirs:
        poses_path = os.path.join(pdir, "poses.npy")
        all_player_poses.append(np.load(poses_path))

    # ---- Build or load BEV + visibility ----
    if args.bev_dir:
        # Legacy path: load from preprocessed files (shared BEV for all players)
        logging.info("Loading BEV + visibility from preprocessed files...")
        player_stems = [s.strip() for s in args.player_stems.split(",") if s.strip()] \
                       if args.player_stems else [None] * N
        bev_map, vis_matrix = load_bev(
            args.bev_dir,
            episode_id=args.episode_id if args.episode_id else None,
            player_stem=player_stems[0] if player_stems[0] else None,
        )
        bev_maps = [bev_map] * N  # same BEV for all players in legacy mode
    else:
        # Self-contained: build per-player BEVs from poses on-the-fly
        logging.info("Building BEV + visibility from poses (self-contained mode)...")
        bev_maps = build_bev_from_poses(
            all_player_poses,
            bev_size=args.bev_size,
            static_bev_path=args.static_bev_path if args.static_bev_path else None,
            navmesh_path=args.navmesh_path if args.navmesh_path else None,
        )
        if args.enable_cross_player:
            vis_matrix = build_visibility_from_poses(all_player_poses)
        else:
            vis_matrix = None

    logging.info(f"BEV map shape: {bev_maps[0].shape} x {N} players")
    if vis_matrix is not None:
        logging.info(f"Visibility matrix shape: {vis_matrix.shape}")

    # Initialize inference engine
    engine = MultiPlayerInference(args)
    engine.load_models()

    # Generate
    videos = engine.generate(
        player_dirs=player_dirs,
        bev_maps=bev_maps,
        visibility_matrix=vis_matrix,
        prompt=args.prompt,
    )

    # Save results
    for i, video in enumerate(videos):
        save_path = os.path.join(args.output_dir, f"player_{i:02d}.mp4")
        save_video_mp4(video, save_path, fps=16)

    # Also save a side-by-side comparison if N <= 4
    if N <= 4 and N > 1:
        try:
            # Stack videos horizontally
            min_frames = min(v.shape[1] for v in videos)
            grid_videos = [v[:, :min_frames] for v in videos]

            # Resize each to half width for side-by-side
            half_w = args.width // 2
            half_h = args.height // 2
            resized = []
            for v in grid_videos:
                v_resized = F.interpolate(
                    v.permute(1, 0, 2, 3),  # [F, 3, H, W]
                    size=(half_h, half_w),
                    mode='bilinear',
                    align_corners=False,
                ).permute(1, 0, 2, 3)  # [3, F, H/2, W/2]
                resized.append(v_resized)

            # Grid layout
            if N == 2:
                grid = torch.cat(resized, dim=3)  # side by side
            elif N == 3:
                blank = torch.zeros_like(resized[0])
                row1 = torch.cat(resized[:2], dim=3)
                row2 = torch.cat([resized[2], blank], dim=3)
                grid = torch.cat([row1, row2], dim=2)
            else:  # N == 4
                row1 = torch.cat(resized[:2], dim=3)
                row2 = torch.cat(resized[2:4], dim=3)
                grid = torch.cat([row1, row2], dim=2)

            grid_path = os.path.join(args.output_dir, "multiview_grid.mp4")
            save_video_mp4(grid, grid_path, fps=16)
        except Exception as e:
            logging.warning(f"Failed to create grid video: {e}")

    logging.info(f"All {N} videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
