"""
Batch Inference & Evaluation for CSGO Fine-tuned LingBot-World.

Runs inference on all clips in the val set (or a subset), computes
quantitative metrics, and generates an evaluation report.

Supports:
  - Stage 1 (single-player) evaluation
  - Stage 2 (multi-player with BEV) evaluation (future)
  - Auto first-frame extraction from video.mp4 if image.jpg missing
  - Metrics: PSNR, SSIM, LPIPS, FVD (optional)

Usage:
    # Full val set evaluation
    torchrun --nproc_per_node=8 eval_batch.py \\
        --ckpt_dir /path/to/base_model \\
        --ft_ckpt_dir /path/to/stage1/final \\
        --dataset_dir /path/to/processed_csgo_v3 \\
        --output_dir /path/to/eval_output \\
        --split val

    # Quick test (first 5 clips)
    torchrun --nproc_per_node=8 eval_batch.py \\
        --ckpt_dir /path/to/base_model \\
        --ft_ckpt_dir /path/to/stage1/final \\
        --dataset_dir /path/to/processed_csgo_v3 \\
        --output_dir /path/to/eval_output \\
        --split val --max_samples 5
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# ============================================================
# Metric computation
# ============================================================

class VideoMetrics:
    """Compute video quality metrics between generated and GT videos."""

    def __init__(self, device="cuda"):
        self.device = device
        self._lpips_model = None

    def _load_lpips(self):
        """Lazy-load LPIPS model."""
        if self._lpips_model is not None:
            return
        try:
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self._lpips_model.eval()
            logging.info("LPIPS model loaded (AlexNet)")
        except ImportError:
            logging.warning("lpips not installed. Run: pip install lpips")
            self._lpips_model = "unavailable"

    def psnr(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute PSNR between two videos.
        Args: pred, gt: [F, H, W, 3] uint8 arrays
        """
        mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * math.log10(255.0 ** 2 / mse)

    def ssim_video(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute average SSIM across frames.
        Args: pred, gt: [F, H, W, 3] uint8 arrays
        """
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            logging.warning("scikit-image not installed for SSIM. Run: pip install scikit-image")
            return float('nan')

        scores = []
        for t in range(min(pred.shape[0], gt.shape[0])):
            s = ssim(pred[t], gt[t], channel_axis=2, data_range=255)
            scores.append(s)
        return float(np.mean(scores))

    def lpips_video(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute average LPIPS across frames.
        Args: pred, gt: [F, H, W, 3] uint8 arrays
        """
        self._load_lpips()
        if self._lpips_model == "unavailable":
            return float('nan')

        scores = []
        for t in range(min(pred.shape[0], gt.shape[0])):
            # Convert to [-1, 1] tensor
            p = torch.from_numpy(pred[t]).permute(2, 0, 1).float() / 127.5 - 1.0
            g = torch.from_numpy(gt[t]).permute(2, 0, 1).float() / 127.5 - 1.0
            p = p.unsqueeze(0).to(self.device)
            g = g.unsqueeze(0).to(self.device)

            with torch.no_grad():
                d = self._lpips_model(p, g).item()
            scores.append(d)

        return float(np.mean(scores))

    def compute_all(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """Compute all available metrics."""
        results = {
            "psnr": self.psnr(pred, gt),
            "ssim": self.ssim_video(pred, gt),
            "lpips": self.lpips_video(pred, gt),
        }
        return results


# ============================================================
# Video I/O utilities
# ============================================================

def read_video_frames(video_path: str, max_frames: int = 81,
                       height: int = 480, width: int = 832) -> np.ndarray:
    """
    Read video frames as numpy array.
    Returns: [F, H, W, 3] uint8 array (RGB)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    return np.stack(frames)


def extract_first_frame(video_path: str, save_path: str,
                         height: int = 480, width: int = 832):
    """Extract and save the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read video: {video_path}")

    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(save_path, frame)
    return save_path


def tensor_to_video_array(tensor: torch.Tensor, height: int = 480,
                           width: int = 832) -> np.ndarray:
    """
    Convert model output tensor to numpy video array.
    Input: [3, F, H, W] tensor in [-1, 1]
    Output: [F, H, W, 3] uint8 array (RGB)
    """
    video = tensor.clamp(-1, 1)
    video = ((video + 1) / 2 * 255).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()  # [F, H, W, 3]
    return video


def save_video_mp4(frames: np.ndarray, save_path: str, fps: int = 16):
    """Save [F, H, W, 3] uint8 RGB frames as MP4."""
    F, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    for t in range(F):
        writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    writer.release()


# ============================================================
# Batch inference runner
# ============================================================

def load_clip_list(dataset_dir: str, split: str = "val") -> list:
    """Load clip list from metadata CSV."""
    csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
    clips = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_dir = os.path.join(dataset_dir, row["clip_path"])
            clips.append({
                "clip_dir": clip_dir,
                "prompt": row.get("prompt", "First-person view of CS:GO competitive gameplay"),
                "episode_id": row.get("episode_id", ""),
                "stem": row.get("stem", ""),
            })
    return clips


def ensure_first_frame(clip_dir: str, height: int = 480, width: int = 832) -> str:
    """Ensure image.jpg exists in clip directory, extract from video if missing."""
    image_path = os.path.join(clip_dir, "image.jpg")
    if os.path.exists(image_path):
        return image_path

    video_path = os.path.join(clip_dir, "video.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Neither image.jpg nor video.mp4 found in {clip_dir}")

    logging.info(f"  Extracting first frame from {video_path}")
    extract_first_frame(video_path, image_path, height, width)
    return image_path


def run_single_inference(wan_i2v, clip_dir: str, prompt: str, args,
                          cfg, save_path: str) -> torch.Tensor:
    """
    Run inference on a single clip using WanI2V.generate().
    Returns video tensor [3, F, H, W].
    """
    image_path = ensure_first_frame(clip_dir, args.height, args.width)
    img = Image.open(image_path).convert("RGB")

    from wan.configs import MAX_AREA_CONFIGS

    video = wan_i2v.generate(
        prompt,
        img,
        action_path=clip_dir + "/",
        max_area=MAX_AREA_CONFIGS[f"{args.height}*{args.width}"],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver="unipc",
        sampling_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=False,
    )

    if video is not None:
        # Save generated video
        from wan.utils.utils import save_video as wan_save_video
        wan_save_video(
            tensor=video[None],
            save_file=save_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    return video


# ============================================================
# Main evaluation pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Inference & Evaluation")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Base model directory")
    parser.add_argument("--lingbot_code_dir", type=str,
                        default="/home/nvme02/lingbot-world/code/lingbot-world")
    parser.add_argument("--ft_ckpt_dir", type=str, default=None,
                        help="Fine-tuned checkpoint dir (dual-model)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA weights path")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Processed dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max clips to evaluate (0 = all)")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip metric computation (only generate videos)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip clips that already have generated videos")

    # Generation config
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=70)
    parser.add_argument("--sample_shift", type=float, default=10.0)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    # Distributed
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)

    args = parser.parse_args()

    # ---- Setup distributed ----
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)

    sys.path.insert(0, args.lingbot_code_dir)

    import torch.distributed as dist
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)

    if args.ulysses_size > 1:
        from wan.distributed.util import init_distributed_group
        init_distributed_group()

    # ---- Build effective checkpoint dir ----
    effective_ckpt_dir = args.ckpt_dir

    if args.lora_path:
        # LoRA: merge into temp dir
        from wan.modules.model import WanModel
        from peft import LoraConfig, inject_adapter_in_model
        import peft.tuners.lora as lora_module

        if rank == 0:
            logging.info("Merging LoRA weights...")

        model = WanModel.from_pretrained(
            args.ckpt_dir, subfolder="low_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )
        lora_state = torch.load(args.lora_path, map_location="cpu", weights_only=True)

        target_modules = set()
        lora_rank = 16
        for key, val in lora_state.items():
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part in ("lora_A", "lora_B"):
                    target_modules.add(".".join(parts[:i]))
                    if part == "lora_A":
                        lora_rank = val.shape[0]
                    break

        config = LoraConfig(r=lora_rank, lora_alpha=lora_rank,
                            target_modules=sorted(list(target_modules)))
        model = inject_adapter_in_model(config, model)

        mapped = {}
        for key, val in lora_state.items():
            if "lora_A.weight" in key and "default" not in key:
                key = key.replace("lora_A.weight", "lora_A.default.weight")
            if "lora_B.weight" in key and "default" not in key:
                key = key.replace("lora_B.weight", "lora_B.default.weight")
            mapped[key] = val
        model.load_state_dict(mapped, strict=False)

        for _name, _mod in model.named_modules():
            if isinstance(_mod, lora_module.Linear):
                _mod.merge()

        tmp_dir = tempfile.mkdtemp(prefix='eval_')
        model.save_pretrained(os.path.join(tmp_dir, "low_noise_model"))
        for item in ["high_noise_model", "Wan2.1_VAE.pth",
                      "models_t5_umt5-xxl-enc-bf16.pth", "google", "configuration.json"]:
            src = os.path.join(args.ckpt_dir, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
        effective_ckpt_dir = tmp_dir
        del model, lora_state

    elif args.ft_ckpt_dir:
        # Dual-model FT: symlink structure
        tmp_dir = tempfile.mkdtemp(prefix='eval_')
        for model_name in ["low_noise_model", "high_noise_model"]:
            src = os.path.join(args.ft_ckpt_dir, model_name)
            dst = os.path.join(tmp_dir, model_name)
            if os.path.exists(src):
                os.symlink(src, dst)
            else:
                os.symlink(os.path.join(args.ckpt_dir, model_name), dst)
        for item in ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                      "google", "configuration.json"]:
            src = os.path.join(args.ckpt_dir, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
        effective_ckpt_dir = tmp_dir

    # ---- Load model ----
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["i2v-A14B"]

    if rank == 0:
        logging.info(f"Loading model from {effective_ckpt_dir}")

    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=effective_ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
    )

    # ---- Load clip list ----
    clips = load_clip_list(args.dataset_dir, args.split)
    if args.max_samples > 0:
        clips = clips[:args.max_samples]

    if rank == 0:
        logging.info(f"Evaluating {len(clips)} {args.split} clips")

    # ---- Distributed clip assignment ----
    # With sequence parallelism (ulysses_size > 1), generate() is a collective
    # operation — all ranks must call it on the SAME clip. Each rank gets the
    # full clip list; only rank 0 saves results and computes metrics.
    # Without SP (ulysses_size == 1), each rank can process different clips.
    use_collective = (args.ulysses_size > 1)
    if use_collective:
        my_clips = clips  # all ranks process every clip together
    else:
        my_clips = clips[rank::world_size]

    if rank == 0:
        logging.info(f"Rank {rank}: processing {len(my_clips)} clips "
                     f"({'collective' if use_collective else 'independent'} mode)")

    # ---- Inference + metrics ----
    metrics_computer = VideoMetrics(device=f"cuda:{local_rank}")
    results = []
    total_time = 0

    for idx, clip in enumerate(my_clips):
        clip_dir = clip["clip_dir"]
        clip_name = os.path.basename(clip_dir)

        gen_path = os.path.join(args.output_dir, "videos", f"{clip_name}_gen.mp4")

        # Skip if already exists (in collective mode, all ranks must agree to skip)
        if args.skip_existing and os.path.exists(gen_path):
            if rank == 0:
                logging.info(f"[{idx+1}/{len(my_clips)}] Skipping {clip_name} (exists)")
            continue

        if rank == 0:
            logging.info(f"[{idx+1}/{len(my_clips)}] Generating {clip_name}...")

        try:
            t0 = time.time()
            # All ranks must call generate together (collective op with SP)
            video_tensor = run_single_inference(
                wan_i2v, clip_dir, clip["prompt"], args, cfg, gen_path,
            )
            gen_time = time.time() - t0
            total_time += gen_time

            # Only rank 0 handles results and metrics
            if use_collective and rank != 0:
                continue

            if video_tensor is None:
                logging.warning(f"  Generation returned None for {clip_name}")
                continue

            logging.info(f"  Generated in {gen_time:.1f}s -> {gen_path}")

            # ---- Compute metrics ----
            clip_result = {
                "clip_name": clip_name,
                "episode_id": clip["episode_id"],
                "gen_time_s": round(gen_time, 1),
            }

            if not args.skip_metrics:
                gt_video_path = os.path.join(clip_dir, "video.mp4")
                if os.path.exists(gt_video_path):
                    gt_frames = read_video_frames(
                        gt_video_path, args.frame_num, args.height, args.width
                    )
                    gen_frames = read_video_frames(
                        gen_path, args.frame_num, args.height, args.width
                    )

                    # Align frame counts
                    min_f = min(len(gt_frames), len(gen_frames))
                    gt_frames = gt_frames[:min_f]
                    gen_frames = gen_frames[:min_f]

                    metrics = metrics_computer.compute_all(gen_frames, gt_frames)
                    clip_result.update(metrics)
                    logging.info(
                        f"  PSNR={metrics['psnr']:.2f} "
                        f"SSIM={metrics['ssim']:.4f} "
                        f"LPIPS={metrics['lpips']:.4f}"
                    )
                else:
                    logging.warning(f"  GT video not found: {gt_video_path}")

            results.append(clip_result)

        except Exception as e:
            logging.error(f"  Error processing {clip_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "clip_name": clip_name,
                "error": str(e),
            })

    # ---- Gather results from all ranks ----
    if world_size > 1:
        dist.barrier()

    # Save per-rank results
    rank_results_path = os.path.join(args.output_dir, f"results_rank{rank}.json")
    with open(rank_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if world_size > 1:
        dist.barrier()

    # ---- Rank 0: aggregate results and write report ----
    if rank == 0:
        all_results = []
        for r in range(world_size):
            rpath = os.path.join(args.output_dir, f"results_rank{r}.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    all_results.extend(json.load(f))

        # Write CSV
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        if all_results:
            fieldnames = list(dict.fromkeys(k for r in all_results for k in r.keys()))
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for r in all_results:
                    writer.writerow(r)

        # Compute aggregates
        valid = [r for r in all_results if "error" not in r and "psnr" in r]

        report = {
            "config": {
                "ckpt_dir": args.ckpt_dir,
                "ft_ckpt_dir": args.ft_ckpt_dir or "",
                "lora_path": args.lora_path or "",
                "split": args.split,
                "num_clips": len(clips),
                "num_evaluated": len(valid),
                "sampling_steps": args.sample_steps,
                "guide_scale": args.guide_scale,
                "frame_num": args.frame_num,
                "resolution": f"{args.height}x{args.width}",
            },
            "aggregate_metrics": {},
            "per_clip": all_results,
        }

        if valid:
            for key in ["psnr", "ssim", "lpips", "gen_time_s"]:
                values = [r[key] for r in valid if key in r and isinstance(r[key], (int, float)) and not math.isnan(r[key])]
                if values:
                    report["aggregate_metrics"][key] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std": round(float(np.std(values)), 4),
                        "min": round(float(np.min(values)), 4),
                        "max": round(float(np.max(values)), 4),
                    }

        # Save report
        report_path = os.path.join(args.output_dir, "eval_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        logging.info("\n" + "=" * 60)
        logging.info("EVALUATION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Split: {args.split}, Clips: {len(clips)}, Evaluated: {len(valid)}")
        if "aggregate_metrics" in report and report["aggregate_metrics"]:
            for metric, stats in report["aggregate_metrics"].items():
                logging.info(f"  {metric:12s}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
                             f"(min={stats['min']:.4f}, max={stats['max']:.4f})")
        logging.info(f"\nResults saved to: {args.output_dir}")
        logging.info(f"  Metrics CSV:  {csv_path}")
        logging.info(f"  Full report:  {report_path}")
        logging.info(f"  Videos:       {os.path.join(args.output_dir, 'videos/')}")
        logging.info("=" * 60)

        # Clean up rank files
        for r in range(world_size):
            rpath = os.path.join(args.output_dir, f"results_rank{r}.json")
            if os.path.exists(rpath):
                os.remove(rpath)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
