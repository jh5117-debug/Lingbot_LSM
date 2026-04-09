"""
LingBot-World CSGO Fine-tuning Training Script (Single-Model)
==============================================================
Trains ONE model (low_noise_model or high_noise_model) per invocation.
For dual-model MoE training, run this script twice via run_train_dual.sh.

Design:
  - Single DeepSpeed engine per run → no dual-engine conflicts
  - ZeRO-3 without CPU offload → 8-10x faster than CPU offload
  - T5 text encoding cached → avoids repeated GPU↔CPU transfers
  - All previously identified bugs fixed

Usage:
    # Train low_noise_model (handles t < 947)
    accelerate launch --config_file accelerate_config_dual.yaml \
        train_lingbot_csgo.py \
        --model_type low \
        --ckpt_dir /home/nvme02/lingbot-world/models/lingbot-world-base-act \
        --dataset_dir /home/nvme02/lingbot-world/datasets/processed_csgo_v3 \
        --output_dir /home/nvme02/lingbot-world/output/dual_ft_v3 \
        --num_epochs 5

    # Train high_noise_model (handles t >= 947)
    accelerate launch --config_file accelerate_config_dual.yaml \
        train_lingbot_csgo.py \
        --model_type high \
        --ckpt_dir /home/nvme02/lingbot-world/models/lingbot-world-base-act \
        --dataset_dir /home/nvme02/lingbot-world/datasets/processed_csgo_v3 \
        --output_dir /home/nvme02/lingbot-world/output/dual_ft_v3 \
        --num_epochs 5
"""

import argparse
import csv
import gc
import logging
import math
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# Dataset
# ============================================================

class CSGODataset(Dataset):
    """Loads preprocessed CSGO clips for LingBot training."""

    def __init__(self, dataset_dir, split="train", num_frames=81,
                 height=480, width=832, repeat=1):
        self.dataset_dir = dataset_dir
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.repeat = repeat

        csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {csv_path}")
        logging.info(f"Loaded {len(self.samples)} {split} samples (x{repeat} repeat)")

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.samples)
        sample = self.samples[idx]
        clip_dir = os.path.join(self.dataset_dir, sample["clip_path"])

        import cv2
        video_path = os.path.join(clip_dir, "video.mp4")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(frame)
        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        video_tensor = torch.stack(frames, dim=1)  # [3, F, H, W]

        poses = np.load(os.path.join(clip_dir, "poses.npy"))
        actions = np.load(os.path.join(clip_dir, "action.npy"))
        intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))

        poses = self._pad_or_truncate(poses, self.num_frames)
        actions = self._pad_or_truncate(actions, self.num_frames)
        intrinsics = self._pad_or_truncate(intrinsics, self.num_frames)

        return {
            "video": video_tensor,
            "prompt": sample["prompt"],
            "poses": torch.from_numpy(poses).float(),
            "actions": torch.from_numpy(actions).float(),
            "intrinsics": torch.from_numpy(intrinsics).float(),
        }

    def _pad_or_truncate(self, arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len]
        rep_shape = (target_len - len(arr),) + (1,) * (arr.ndim - 1)
        pad = np.tile(arr[-1:], rep_shape)
        return np.concatenate([arr, pad], axis=0)


# ============================================================
# Training Module
# ============================================================

class LingBotTrainer:
    """Handles model loading, control signal preparation, and training loop."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        self.boundary = 0.947
        self.num_train_timesteps = 1000
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.shift = 10.0  # Wan/LingBot sigma schedule shift

        # Pre-compute shifted sigma schedule
        sigmas_linear = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.shift * sigmas_linear / (1 + (self.shift - 1) * sigmas_linear)
        self.timesteps_schedule = self.sigmas * self.num_train_timesteps

        # Split timestep indices by model boundary
        max_timestep = self.boundary * self.num_train_timesteps  # 947
        self.low_noise_indices = torch.where(self.timesteps_schedule < max_timestep)[0]
        self.high_noise_indices = torch.where(self.timesteps_schedule >= max_timestep)[0]
        logging.info(f"Sigma schedule: shift={self.shift}, "
                     f"{len(self.low_noise_indices)} low-noise timesteps (t < {max_timestep}), "
                     f"{len(self.high_noise_indices)} high-noise timesteps (t >= {max_timestep})")

        # Compute training weights (Gaussian-like centered at t=500)
        x = self.timesteps_schedule
        steps = self.num_train_timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        self.training_weights = y_shifted * (steps / y_shifted.sum())

        # Normalize weights per range so each model gets well-scaled gradients
        low_w = self.training_weights[self.low_noise_indices]
        self.low_noise_weights = low_w / low_w.mean()
        high_w = self.training_weights[self.high_noise_indices]
        self.high_noise_weights = high_w / high_w.mean()

        # T5 encoding cache (prompt string -> tensor list)
        self._t5_cache = {}

    def load_model(self, device, model_type="low"):
        """Load a single model and shared components (VAE, T5)."""
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

        # Check if fine-tuned checkpoint exists (for resuming)
        load_dir = ckpt_dir
        if self.args.resume_from:
            resume_model_dir = os.path.join(self.args.resume_from, subfolder)
            if os.path.isdir(resume_model_dir):
                load_dir = self.args.resume_from
                logging.info(f"Resuming from checkpoint: {resume_model_dir}")

        logging.info(f"Loading {subfolder} from {load_dir}...")
        model = WanModel.from_pretrained(
            load_dir, subfolder=subfolder,
            torch_dtype=torch.bfloat16, control_type="act",
        )

        # Extend control signal: rays_d(3) + action(8) = 11ch (base model has 7ch)
        new_control_dim = 3 + self.args.action_dim  # default: 3 + 8 = 11
        model = _extend_control_embedding(model, new_control_dim, subfolder)

        model.train()
        wancamctrl = model.patch_embedding_wancamctrl
        logging.info(f"{subfolder} patch_embedding_wancamctrl: "
                     f"Linear({wancamctrl.in_features}, {wancamctrl.out_features})")

        logging.info("Loading VAE...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
            device=self.device,
        )

        logging.info("Loading T5 text encoder...")
        self.t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(ckpt_dir, "google", "umt5-xxl"),
        )

        return model

    def freeze_non_trainable(self, model, name="model"):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(f"Full FT ({name}): {trainable:,} trainable / {total:,} total params")

    @torch.no_grad()
    def encode_video(self, video_tensor):
        """video_tensor: [3, F, H, W] -> latent [16, lat_f, lat_h, lat_w]"""
        latent = self.vae.encode([video_tensor.to(self.device)])[0]
        torch.cuda.empty_cache()
        return latent

    @torch.no_grad()
    def encode_text(self, prompt):
        """prompt string -> list of text embedding tensors (cached)."""
        if prompt in self._t5_cache:
            return [t.to(self.device) for t in self._t5_cache[prompt]]

        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()

        # Cache on CPU to avoid GPU memory leak
        self._t5_cache[prompt] = [t.cpu() for t in context]
        return [t.to(self.device) for t in context]

    @torch.no_grad()
    def prepare_y(self, video_tensor, latent):
        """
        Prepare conditional input y (mask + first frame VAE encoding).
        Must match image2video.py lines 392-401.
        """
        lat_f, lat_h, lat_w = latent.shape[1], latent.shape[2], latent.shape[3]
        F_total = video_tensor.shape[1]
        h, w = video_tensor.shape[2], video_tensor.shape[3]

        first_frame = video_tensor[:, 0:1, :, :]  # [3, 1, H, W]
        zeros = torch.zeros(3, F_total - 1, h, w, device=video_tensor.device)
        vae_input = torch.concat([first_frame, zeros], dim=1)  # [3, F, H, W]
        y_latent = self.vae.encode([vae_input.to(self.device)])[0]

        msk = torch.ones(1, F_total, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, lat_f, lat_h, lat_w]

        y = torch.concat([msk, y_latent])  # [20, lat_f, lat_h, lat_w]
        return y

    @torch.no_grad()
    def prepare_control_signal(self, poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w):
        """
        Prepare dit_cond_dict for act mode.
        Act mode: rays_d(3) + wasd(4) = 7 control channels, control_dim=7
        """
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
        )  # [lat_f, h, w, 3]

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
        """Single training step with Flow Matching loss."""
        video = batch["video"].to(self.device)
        prompt = batch["prompt"]
        poses = batch["poses"]
        actions = batch["actions"]
        intrinsics = batch["intrinsics"]

        h, w = video.shape[2], video.shape[3]

        with torch.no_grad():
            video_latent = self.encode_video(video)

        lat_f, lat_h, lat_w = video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
        seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        with torch.no_grad():
            context = self.encode_text(prompt)
            y = self.prepare_y(video, video_latent)

        # [FIX mem] free video tensor — no longer needed after prepare_y
        del video

        # [FIX bug#5] control signal wrapped in no_grad
        with torch.no_grad():
            dit_cond_dict = self.prepare_control_signal(
                poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
            )

        # Sample timestep from the appropriate range
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

        # Flow Matching: add noise
        noise = torch.randn_like(video_latent)
        noisy_latent = (1.0 - sigma) * video_latent + sigma * noise

        # Target: velocity = noise - clean
        target = noise - video_latent

        # [FIX mem] free intermediates before DiT forward
        del noise, video_latent
        torch.cuda.empty_cache()

        # [FIX bug#4] removed: noisy_latent.requires_grad_(True)

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(
                [noisy_latent],
                t=t,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond_dict,
            )[0]

        # Loss (exclude first temporal frame which is conditioned)
        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        loss = F.mse_loss(pred_rest.float(), target_rest.float())
        loss = loss * training_weight
        return loss


# ============================================================
# Control dimension extension helper
# ============================================================

def _extend_control_embedding(model, new_control_dim, name="model"):
    """
    Extend patch_embedding_wancamctrl to support more action channels.
    Base model: control_dim=7 (rays_d 3 + WASD 4), Linear(1792, 5120).
    Extended:   control_dim=11 (rays_d 3 + action 8), Linear(2816, 5120).

    Pretrained weights for original 7 channels are preserved.
    New channels are zero-initialized (no effect at start, learned during training).
    """
    old_linear = model.patch_embedding_wancamctrl
    old_in = old_linear.in_features
    out_dim = old_linear.out_features
    ps = model.patch_size
    factor = 64 * ps[0] * ps[1] * ps[2]  # spatial packing factor
    old_control_dim = old_in // factor
    new_in = new_control_dim * factor

    if old_control_dim >= new_control_dim:
        return model

    logging.info(f"{name}: extending control_dim {old_control_dim} -> {new_control_dim} "
                 f"(Linear {old_in} -> {new_in}, +{new_in - old_in} input features)")

    new_linear = torch.nn.Linear(new_in, out_dim, dtype=old_linear.weight.dtype,
                                 device=old_linear.weight.device)
    torch.nn.init.zeros_(new_linear.weight)
    torch.nn.init.zeros_(new_linear.bias)
    new_linear.weight.data[:, :old_in] = old_linear.weight.data
    new_linear.bias.data = old_linear.bias.data
    model.patch_embedding_wancamctrl = new_linear
    return model


# ============================================================
# Gradient checkpointing helper
# ============================================================

def apply_gradient_checkpointing(model, model_name="model"):
    """Apply gradient checkpointing to DiT blocks of a WanModel."""
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    from functools import wraps

    patched = 0
    block_container = None
    for attr in ['blocks', 'layers', 'transformer_blocks']:
        if hasattr(model, attr):
            block_container = getattr(model, attr)
            break
    if block_container is None:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 10:
                block_container = mod
                break

    if block_container is not None:
        for block in block_container:
            if not any(p.requires_grad for p in block.parameters()):
                continue
            orig_forward = block.forward
            def _make_ckpt_fn(fn):
                @wraps(fn)
                def _ckpt_forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
                    return torch_checkpoint(fn, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict, use_reentrant=False)
                return _ckpt_forward
            block.forward = _make_ckpt_fn(orig_forward)
            patched += 1
        logging.info(f"Gradient checkpointing ({model_name}): patched {patched} DiT blocks")
    else:
        logging.warning(f"Gradient checkpointing ({model_name}): could not find DiT blocks, skipping")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LingBot-World CSGO Fine-tuning (Single-Model)")
    parser.add_argument("--model_type", type=str, required=True, choices=["low", "high"],
                        help="Which model to train: 'low' (t<947) or 'high' (t>=947)")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Base model directory (lingbot-world-base-act)")
    parser.add_argument("--lingbot_code_dir", type=str,
                        default="/home/nvme02/lingbot-world/code/lingbot-world")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a previous checkpoint directory")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--action_dim", type=int, default=8,
                        help="Action dimensions in action.npy (4=WASD only, 8=WASD+jump/crouch/fire/walk)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    model_type = args.model_type
    subfolder = "low_noise_model" if model_type == "low" else "high_noise_model"

    import accelerate
    from accelerate.utils import DataLoaderConfiguration
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Training {subfolder} (model_type={model_type})")
        logging.info(f"Args: {args}")

    # Load single model + VAE + T5
    trainer = LingBotTrainer(args)
    model = trainer.load_model(accelerator.device, model_type)

    # Full fine-tuning (all params trainable)
    trainer.freeze_non_trainable(model, subfolder)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        apply_gradient_checkpointing(model, subfolder)

    # Dataset and dataloader
    dataset = CSGODataset(
        args.dataset_dir, split="train",
        num_frames=args.num_frames, height=args.height, width=args.width,
        repeat=args.dataset_repeat,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # [FIX bug#3] T_max based on optimizer steps, accounting for distributed
    # len(dataloader) before prepare() returns TOTAL batches, not per-GPU
    iters_per_epoch = math.ceil(len(dataset) / accelerator.num_processes)
    steps_per_epoch = iters_per_epoch // args.gradient_accumulation_steps
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-6
    )

    # Prepare with accelerator — single DeepSpeed engine, no conflicts
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Pre-warm T5 cache: encode a dummy prompt to avoid cold start
    if accelerator.is_main_process:
        logging.info("Pre-warming T5 cache...")
    _ = trainer.encode_text("First-person view of CS:GO competitive gameplay")

    # Move T5 to CPU permanently after cache warm-up to free GPU memory
    trainer.t5.model.cpu()
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logging.info(f"Training config: {args.num_epochs} epochs, "
                     f"{len(dataloader)} iters/epoch, "
                     f"grad_accum={args.gradient_accumulation_steps}, "
                     f"optimizer_steps/epoch={steps_per_epoch}, "
                     f"total_optimizer_steps={total_steps}")

    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, disable=not accelerator.is_main_process,
                        desc=f"Epoch {epoch+1}/{args.num_epochs} [{subfolder}]")

        for batch in progress:
            with accelerator.accumulate(model):
                loss = trainer.training_step(
                    accelerator.unwrap_model(model), batch, model_type
                )
                accelerator.backward(loss)

                # [FIX bug#2] use model.parameters() not stale param list
                # [FIX bug#1] only clip/step scheduler on sync_gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} [{subfolder}] "
                         f"- avg_loss: {avg_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_checkpoint(accelerator, model, args, subfolder, f"epoch_{epoch+1}")

    # Final save
    save_checkpoint(accelerator, model, args, subfolder, "final")
    if accelerator.is_main_process:
        logging.info(f"Training complete! {subfolder} saved to {args.output_dir}/final/{subfolder}")


def save_checkpoint(accelerator, model, args, subfolder, tag):
    """Save model checkpoint compatible with WanModel.from_pretrained()."""
    save_dir = os.path.join(args.output_dir, tag)
    model_dir = os.path.join(save_dir, subfolder)

    if accelerator.is_main_process:
        os.makedirs(model_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # save_16bit_model: collective op, gathers ZeRO-3 shards, rank 0 writes
    model.save_16bit_model(model_dir, "diffusion_pytorch_model.bin")

    if accelerator.is_main_process:
        # Save config.json for from_pretrained() compatibility
        unwrapped = accelerator.unwrap_model(model)
        # Update control_dim in config so from_pretrained creates correct Linear size
        new_control_dim = 3 + args.action_dim
        if new_control_dim != 7:
            unwrapped.config["control_dim"] = new_control_dim
        unwrapped.save_config(model_dir)

        # Verify
        saved_path = os.path.join(model_dir, "diffusion_pytorch_model.bin")
        if os.path.exists(saved_path):
            sd = torch.load(saved_path, map_location="cpu", weights_only=True)
            n_empty = sum(1 for v in sd.values() if v.numel() == 0)
            size_mb = os.path.getsize(saved_path) / (1024 * 1024)
            logging.info(f"Saved {subfolder} -> {model_dir} "
                         f"({len(sd)} params, {n_empty} empty, {size_mb:.0f} MB)")
            if n_empty > 0:
                logging.error(f"WARNING: {n_empty} parameters have empty tensors!")
            del sd

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
