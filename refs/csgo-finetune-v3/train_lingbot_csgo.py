"""
LingBot-World CSGO Fine-tuning Training Script (Dual-Model MoE)
================================================================
Trains both high_noise_model (t >= 947) and low_noise_model (t < 947)
using alternating epochs: even epochs train low_noise_model, odd epochs
train high_noise_model. Both models use control_type='act' (7-channel).

Usage:
    accelerate launch --config_file accelerate_config_zero2.yaml \
        train_lingbot_csgo.py \
        --ckpt_dir /home/nvme02/lingbot-world/models/lingbot-world-base-act/ \
        --dataset_dir /home/nvme02/lingbot-world/datasets/csgo_processed_v3/ \
        --output_dir /home/nvme02/lingbot-world/output/csgo_dual_ft/ \
        --lora_rank 0 \
        --learning_rate 1e-5 \
        --num_epochs 10 \
        --height 480 --width 832 --num_frames 81
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
        # Without this, high_noise_model (t>=947) would get near-zero weights
        # since the Gaussian peaks at t=500
        low_w = self.training_weights[self.low_noise_indices]
        self.low_noise_weights = low_w / low_w.mean()
        high_w = self.training_weights[self.high_noise_indices]
        self.high_noise_weights = high_w / high_w.mean()

    def load_models(self, device):
        """Load all model components (both high and low noise models)."""
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

        logging.info("Loading low_noise_model (trainable)...")
        self.low_noise_model = WanModel.from_pretrained(
            ckpt_dir, subfolder="low_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )
        self.low_noise_model.train()
        wancamctrl = self.low_noise_model.patch_embedding_wancamctrl
        logging.info(f"low_noise_model patch_embedding_wancamctrl: "
                     f"Linear({wancamctrl.in_features}, {wancamctrl.out_features})")

        logging.info("Loading high_noise_model (trainable)...")
        self.high_noise_model = WanModel.from_pretrained(
            ckpt_dir, subfolder="high_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )
        self.high_noise_model.train()
        wancamctrl_h = self.high_noise_model.patch_embedding_wancamctrl
        logging.info(f"high_noise_model patch_embedding_wancamctrl: "
                     f"Linear({wancamctrl_h.in_features}, {wancamctrl_h.out_features})")

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

        return self.low_noise_model, self.high_noise_model

    def setup_lora(self, model, lora_rank, lora_target_modules):
        """Apply LoRA to the model."""
        from peft import LoraConfig, inject_adapter_in_model

        if lora_target_modules:
            target_modules = lora_target_modules.split(",")
        else:
            target_modules = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    for pattern in ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                                    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                                    "ffn.0", "ffn.2",
                                    "cam_injector_layer1", "cam_injector_layer2",
                                    "cam_scale_layer", "cam_shift_layer"]:
                        if pattern in name:
                            target_modules.append(name)
                            break

        logging.info(f"LoRA target modules ({len(target_modules)}): {target_modules[:5]}...")

        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)

        for param in model.parameters():
            if param.requires_grad:
                param.data = param.to(torch.bfloat16)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(f"LoRA: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)")
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
        """prompt string -> list of text embedding tensors"""
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()
        return [t.to(self.device) for t in context]

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

        noisy_latent = noisy_latent.requires_grad_(True)
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
    parser = argparse.ArgumentParser(description="LingBot-World CSGO Fine-tuning (Dual-Model MoE)")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--lingbot_code_dir", type=str,
                        default="/home/nvme02/lingbot-world/code/lingbot-world")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank. 0 = full fine-tuning")
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    import accelerate
    from accelerate.utils import DataLoaderConfiguration
    accelerator = accelerate.Accelerator(
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Args: {args}")

    # Load both models
    trainer = LingBotTrainer(args)
    low_noise_model, high_noise_model = trainer.load_models(accelerator.device)

    # Setup LoRA or full fine-tuning for both models
    if args.lora_rank > 0:
        low_noise_model = trainer.setup_lora(low_noise_model, args.lora_rank, args.lora_target_modules)
        high_noise_model = trainer.setup_lora(high_noise_model, args.lora_rank, args.lora_target_modules)
    else:
        trainer.freeze_non_trainable(low_noise_model, "low_noise_model")
        trainer.freeze_non_trainable(high_noise_model, "high_noise_model")

    # Apply gradient checkpointing to both models
    if args.gradient_checkpointing:
        apply_gradient_checkpointing(low_noise_model, "low_noise_model")
        apply_gradient_checkpointing(high_noise_model, "high_noise_model")

    # Dataset and dataloader
    dataset = CSGODataset(
        args.dataset_dir, split="train",
        num_frames=args.num_frames, height=args.height, width=args.width,
        repeat=args.dataset_repeat,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    # Separate optimizers for each model
    low_params = [p for p in low_noise_model.parameters() if p.requires_grad]
    high_params = [p for p in high_noise_model.parameters() if p.requires_grad]

    low_optimizer = torch.optim.AdamW(low_params, lr=args.learning_rate,
                                       weight_decay=args.weight_decay)
    high_optimizer = torch.optim.AdamW(high_params, lr=args.learning_rate,
                                        weight_decay=args.weight_decay)

    total_steps_per_model = (args.num_epochs // 2) * len(dataloader)
    low_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        low_optimizer, T_max=max(total_steps_per_model, 1), eta_min=1e-6
    )
    high_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        high_optimizer, T_max=max(total_steps_per_model, 1), eta_min=1e-6
    )

    # Prepare with accelerator — each model gets its own DeepSpeed engine
    low_noise_model, low_optimizer, dataloader, low_scheduler = accelerator.prepare(
        low_noise_model, low_optimizer, dataloader, low_scheduler
    )
    high_noise_model, high_optimizer, high_scheduler = accelerator.prepare(
        high_noise_model, high_optimizer, high_scheduler
    )

    # Training loop: alternate epochs between models
    global_step = 0
    for epoch in range(args.num_epochs):
        # Even epochs (0, 2, 4, ...) -> low_noise_model
        # Odd epochs (1, 3, 5, ...) -> high_noise_model
        if epoch % 2 == 0:
            current_model = low_noise_model
            current_optimizer = low_optimizer
            current_scheduler = low_scheduler
            current_params = low_params
            model_type = "low"
            model_name = "low_noise_model"
        else:
            current_model = high_noise_model
            current_optimizer = high_optimizer
            current_scheduler = high_scheduler
            current_params = high_params
            model_type = "high"
            model_name = "high_noise_model"

        current_model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, disable=not accelerator.is_main_process,
                        desc=f"Epoch {epoch+1}/{args.num_epochs} [{model_name}]")

        for batch in progress:
            with accelerator.accumulate(current_model):
                loss = trainer.training_step(
                    accelerator.unwrap_model(current_model), batch, model_type
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(current_params, args.max_grad_norm)
                current_optimizer.step()
                current_scheduler.step()
                current_optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_scheduler.get_last_lr()[0]:.2e}",
                model=model_type,
            )

            if args.save_steps and global_step % args.save_steps == 0:
                save_checkpoint(accelerator, low_noise_model, high_noise_model, args,
                                f"step_{global_step}")

        avg_loss = epoch_loss / max(num_batches, 1)
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} [{model_name}] - avg_loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_checkpoint(accelerator, low_noise_model, high_noise_model, args,
                            f"epoch_{epoch+1}")

    save_checkpoint(accelerator, low_noise_model, high_noise_model, args, "final")
    if accelerator.is_main_process:
        logging.info("Training complete!")


def save_checkpoint(accelerator, low_noise_model, high_noise_model, args, tag):
    """Save both models' checkpoints in the original directory structure."""
    save_dir = os.path.join(args.output_dir, tag)

    if args.lora_rank > 0:
        # LoRA: only main process needs to save, no collective operation
        if accelerator.is_main_process:
            for model, name in [(low_noise_model, "low_noise_model"),
                                 (high_noise_model, "high_noise_model")]:
                model_dir = os.path.join(save_dir, name)
                os.makedirs(model_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                lora_state_dict = {
                    n: p.data.cpu()
                    for n, p in unwrapped.named_parameters()
                    if p.requires_grad
                }
                torch.save(lora_state_dict, os.path.join(model_dir, "lora_weights.pth"))
                logging.info(f"Saved LoRA [{name}] ({len(lora_state_dict)} params) -> {model_dir}")
    else:
        # ZeRO-3: use DeepSpeed's native save_16bit_model
        if accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        for model, name in [(low_noise_model, "low_noise_model"),
                             (high_noise_model, "high_noise_model")]:
            model_dir = os.path.join(save_dir, name)
            if accelerator.is_main_process:
                os.makedirs(model_dir, exist_ok=True)
            accelerator.wait_for_everyone()

            # save_16bit_model is a collective op: all ranks gather, rank 0 writes
            model.save_16bit_model(model_dir, "diffusion_pytorch_model.bin")

            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_config(model_dir)
                # Verify saved weights
                saved_path = os.path.join(model_dir, "diffusion_pytorch_model.bin")
                sd = torch.load(saved_path, map_location="cpu", weights_only=True)
                n_empty = sum(1 for v in sd.values() if v.numel() == 0)
                logging.info(f"Saved {name} -> {model_dir} ({len(sd)} params, {n_empty} empty)")
                if n_empty > 0:
                    logging.error(f"WARNING: {n_empty} parameters have empty tensors in {name}!")
                del sd

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
