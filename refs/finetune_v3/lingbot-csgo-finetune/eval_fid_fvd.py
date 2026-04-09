"""
FID / FVD Evaluation for CSGO Fine-tuned LingBot-World.

FID  — Fréchet Inception Distance (clean-fid, InceptionV3 features)
FVD  — Fréchet Video Distance (TorchScript I3D, VideoGPT variant)

Compares generated videos (*_gen.mp4 from eval_batch.py) against
ground-truth videos (video.mp4 in clip directories).

Usage:
    python eval_fid_fvd.py \
        --gen_dir   /path/to/eval_output/videos \
        --real_dir  /path/to/processed_csgo_v3/val/clips \
        --output_dir /path/to/fid_fvd_output \
        --device cuda:0

    # Skip FVD (faster, for quick checks):
    python eval_fid_fvd.py --gen_dir ... --real_dir ... --skip_fvd

    # Use pre-downloaded I3D weights:
    python eval_fid_fvd.py --gen_dir ... --real_dir ... \
        --i3d_path /path/to/i3d_torchscript.pt
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I3D weight URL (VideoGPT variant, TorchScript)
# ---------------------------------------------------------------------------
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"


# ===========================================================================
# Video I/O helpers
# ===========================================================================

def read_video_frames(path: str, max_frames: int = 0) -> np.ndarray:
    """Read video -> uint8 np.ndarray [T, H, W, 3] RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize [T, H, W, 3] to [T, size, size, 3]."""
    out = []
    for f in frames:
        out.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(out, axis=0)


def centre_crop_clip(frames: np.ndarray, clip_len: int) -> np.ndarray:
    """Centre-sample exactly clip_len frames from a video."""
    T = len(frames)
    if T == clip_len:
        return frames
    if T < clip_len:
        pad = np.repeat(frames[[-1]], clip_len - T, axis=0)
        return np.concatenate([frames, pad], axis=0)
    start = (T - clip_len) // 2
    return frames[start: start + clip_len]


# ===========================================================================
# FID  (uses clean-fid library)
# ===========================================================================

def compute_fid(real_frames_list, gen_frames_list, device: str,
                frame_stride: int = 4) -> float:
    """
    Compute FID between two lists of [T,H,W,3] uint8 frame arrays.
    Sub-samples frames at `frame_stride`, resizes to 299x299 for InceptionV3.
    """
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        log.warning("clean-fid not installed (pip install clean-fid). Skipping FID.")
        return None

    from PIL import Image as PILImage

    real_dir = tempfile.mkdtemp(prefix="fid_real_")
    gen_dir = tempfile.mkdtemp(prefix="fid_gen_")
    try:
        # Save real frames as PNGs
        idx = 0
        for frames in real_frames_list:
            sampled = frames[::frame_stride]
            resized = resize_frames(sampled, 299)
            for f in resized:
                PILImage.fromarray(f).save(os.path.join(real_dir, f"{idx:06d}.png"))
                idx += 1
        real_count = idx

        # Save generated frames as PNGs
        idx = 0
        for frames in gen_frames_list:
            sampled = frames[::frame_stride]
            resized = resize_frames(sampled, 299)
            for f in resized:
                PILImage.fromarray(f).save(os.path.join(gen_dir, f"{idx:06d}.png"))
                idx += 1
        gen_count = idx

        if real_count == 0 or gen_count == 0:
            log.warning("No frames collected for FID.")
            return None

        log.info(f"FID: {real_count} real frames, {gen_count} gen frames")
        score = cleanfid.compute_fid(real_dir, gen_dir, device=device, verbose=False)
        return float(score)
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(gen_dir, ignore_errors=True)


# ===========================================================================
# FVD  (I3D TorchScript model)
# ===========================================================================

def download_i3d(save_path: str):
    """Download I3D TorchScript weights if not present."""
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    log.info(f"Downloading I3D weights -> {save_path} ...")
    urllib.request.urlretrieve(I3D_URL, save_path)
    log.info("Download complete.")


def load_i3d(i3d_path: str, device: str):
    """Load I3D TorchScript model."""
    if not os.path.exists(i3d_path):
        download_i3d(i3d_path)
    model = torch.jit.load(i3d_path, map_location=device)
    model.eval()
    return model


def preprocess_clip_for_i3d(frames: np.ndarray, clip_len: int = 16,
                            spatial_size: int = 224) -> torch.Tensor:
    """
    Prepare one video clip for I3D.
    Input:  uint8 [T, H, W, 3] RGB
    Output: float32 [1, T, H, W, 3] in [-1, 1]
    """
    frames = centre_crop_clip(frames, clip_len)
    resized = resize_frames(frames, spatial_size)
    tensor = torch.from_numpy(resized).float() / 127.5 - 1.0  # [-1, 1]
    return tensor.unsqueeze(0)  # [1, T, H, W, 3]


@torch.no_grad()
def extract_i3d_features(video_frames_list, i3d_model, device: str,
                         clip_len: int = 16, spatial_size: int = 224,
                         batch_size: int = 8) -> np.ndarray:
    """
    Extract I3D features for a list of video frame arrays.
    Each video is centre-cropped to clip_len frames.
    Returns np.ndarray [N, D].
    """
    clips = []
    for frames in video_frames_list:
        if len(frames) == 0:
            continue
        clip = preprocess_clip_for_i3d(frames, clip_len, spatial_size)
        clips.append(clip)

    if not clips:
        return np.zeros((0, 400))

    all_feats = []
    for i in range(0, len(clips), batch_size):
        batch = torch.cat(clips[i: i + batch_size], dim=0).to(device)
        feats = i3d_model(batch)  # [B, D]
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Frechet distance between two multivariate Gaussians."""
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_fvd(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    """Compute FVD from pre-extracted I3D features."""
    if len(real_feats) < 2 or len(gen_feats) < 2:
        log.warning("Too few samples for FVD (need >= 2). Returning inf.")
        return float("inf")
    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


# ===========================================================================
# Data discovery
# ===========================================================================

def find_gen_videos(gen_dir: str):
    """
    Find generated videos.
    Looks for *_gen.mp4 (eval_batch.py output), falls back to any *.mp4.
    """
    gen_dir = Path(gen_dir)
    videos = sorted(gen_dir.rglob("*_gen.mp4"))
    if not videos:
        videos = sorted(gen_dir.rglob("*.mp4"))
    return [str(v) for v in videos]


def find_real_videos(real_dir: str):
    """
    Find GT videos.
    Expects:  real_dir/<clip_id>/video.mp4
    """
    real_dir = Path(real_dir)
    videos = sorted(real_dir.rglob("video.mp4"))
    return [str(v) for v in videos]


def match_gen_to_real(gen_paths, real_paths):
    """
    Match generated videos to real videos by clip stem name.
    gen name pattern: <clip_stem>_gen.mp4
    real name pattern: <clip_stem>/video.mp4

    Returns list of (gen_path, real_path) tuples.
    """
    # Build lookup: clip_stem -> real_path
    real_lookup = {}
    for rp in real_paths:
        stem = Path(rp).parent.name  # clip directory name
        real_lookup[stem] = rp

    pairs = []
    for gp in gen_paths:
        gname = Path(gp).stem  # e.g. "Ep_000005_team_2_player_0000_clip0020_gen"
        # Remove _gen suffix
        if gname.endswith("_gen"):
            clip_stem = gname[:-4]
        else:
            clip_stem = gname

        if clip_stem in real_lookup:
            pairs.append((gp, real_lookup[clip_stem]))
        else:
            log.debug(f"No matching real video for {gname}")

    return pairs


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="FID / FVD evaluation for CSGO video generation")
    p.add_argument("--gen_dir", required=True,
                   help="Directory containing generated videos (*_gen.mp4)")
    p.add_argument("--real_dir", required=True,
                   help="GT clip directory (each clip has video.mp4)")
    p.add_argument("--output_dir", required=True,
                   help="Where to save eval_fid_fvd_report.json")
    p.add_argument("--i3d_path", default="i3d_torchscript.pt",
                   help="Path to I3D TorchScript weights (auto-downloaded if missing)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clip_len", type=int, default=16,
                   help="Number of frames per FVD clip (default: 16)")
    p.add_argument("--num_clips", type=int, default=0,
                   help="Max number of video pairs to use (0 = all)")
    p.add_argument("--frame_stride", type=int, default=4,
                   help="Frame sub-sampling stride for FID (default: 4)")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--skip_fvd", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Discover videos ----
    gen_paths = find_gen_videos(args.gen_dir)
    real_paths = find_real_videos(args.real_dir)
    log.info(f"Found {len(gen_paths)} generated videos, {len(real_paths)} real videos")

    # Match by clip stem
    pairs = match_gen_to_real(gen_paths, real_paths)
    log.info(f"Matched {len(pairs)} gen-real pairs")

    if not pairs:
        # Fallback: just pair by sorted order (truncate to min)
        n = min(len(gen_paths), len(real_paths))
        if n == 0:
            log.error("No videos found. Check --gen_dir and --real_dir.")
            return
        log.warning(f"No stem matches found. Falling back to sorted-order pairing ({n} pairs)")
        pairs = list(zip(gen_paths[:n], real_paths[:n]))

    if args.num_clips > 0:
        pairs = pairs[:args.num_clips]

    n = len(pairs)
    log.info(f"Using {n} pairs for evaluation")

    report = {
        "num_clips": n,
        "gen_dir": args.gen_dir,
        "real_dir": args.real_dir,
        "frame_stride": args.frame_stride,
        "clip_len": args.clip_len,
    }

    # ---- Load frames ----
    log.info("Loading video frames ...")
    t0 = time.time()
    gen_frames_list = [read_video_frames(gp) for gp, _ in pairs]
    real_frames_list = [read_video_frames(rp) for _, rp in pairs]
    log.info(f"Frame loading done in {time.time()-t0:.1f}s "
             f"(gen: {sum(len(f) for f in gen_frames_list)} frames, "
             f"real: {sum(len(f) for f in real_frames_list)} frames)")

    # ---- FID ----
    if not args.skip_fid:
        log.info("Computing FID ...")
        t0 = time.time()
        fid_score = compute_fid(real_frames_list, gen_frames_list,
                                device=args.device, frame_stride=args.frame_stride)
        elapsed = time.time() - t0
        if fid_score is not None:
            log.info(f"FID = {fid_score:.2f}  ({elapsed:.1f}s)")
        report["fid"] = fid_score
        report["fid_time_s"] = round(elapsed, 1)
    else:
        report["fid"] = None

    # ---- FVD ----
    if not args.skip_fvd:
        log.info("Loading I3D model ...")
        i3d = load_i3d(args.i3d_path, args.device)

        log.info("Extracting I3D features (generated) ...")
        t0 = time.time()
        gen_feats = extract_i3d_features(gen_frames_list, i3d, args.device,
                                         clip_len=args.clip_len)
        log.info("Extracting I3D features (real) ...")
        real_feats = extract_i3d_features(real_frames_list, i3d, args.device,
                                          clip_len=args.clip_len)
        fvd_score = compute_fvd(real_feats, gen_feats)
        elapsed = time.time() - t0
        log.info(f"FVD = {fvd_score:.2f}  ({elapsed:.1f}s)")
        report["fvd"] = fvd_score
        report["fvd_time_s"] = round(elapsed, 1)
        report["num_feats_gen"] = len(gen_feats)
        report["num_feats_real"] = len(real_feats)
    else:
        report["fvd"] = None

    # ---- Save report ----
    out_path = os.path.join(args.output_dir, "eval_fid_fvd_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved -> {out_path}")

    # ---- Print summary ----
    print("\n" + "=" * 50)
    print("FID / FVD EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Clips evaluated : {n}")
    if report.get("fid") is not None:
        print(f"  FID             : {report['fid']:.2f}")
    if report.get("fvd") is not None:
        print(f"  FVD             : {report['fvd']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
