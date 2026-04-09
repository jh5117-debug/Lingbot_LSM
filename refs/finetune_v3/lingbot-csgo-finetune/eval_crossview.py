"""
Cross-View Consistency Evaluation for Stage 2 Multi-View Video Generation.

Evaluates geometric consistency between generated videos from different players
in the same episode by comparing:
  1. Epipolar Error: Keypoint matches between two views should lie on epipolar lines.
  2. Feature Matching Rate: ORB inlier ratio between overlapping views.
  3. Depth Reprojection PSNR: Reproject pixels from view A to view B using GT depth,
     compare reprojected appearance to generated view B.

Depth encoding:
  The _depth.mkv files use ffv1 lossless codec, pixel format rgb24.
  16-bit depth is packed as: depth_16bit = R * 256 + G (R=high byte, G=low byte)
  Depth range: 0–65535 raw units, covering 0–2048 game units.

Usage:
    python eval_crossview.py \\
        --stage2_gen_dir /path/to/stage2_inference/phase2b \\
        --stage1_gen_dir /path/to/eval_stage1/videos \\
        --dataset_dir /path/to/processed_csgo_v3 \\
        --raw_data_dir /path/to/raw_csgo_v3/.../train \\
        --output_dir /path/to/crossview_results \\
        --split val \\
        --max_episode_pairs 50
"""

import argparse
import csv
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# Depth decoding
# ============================================================

def decode_depth_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Decode 16-bit depth from ffv1/rgb24 frame.

    The depth is packed as: depth_16 = R * 256 + G (R=high byte, G=low byte).
    OpenCV reads as BGR, so channel order is B=0, G=1, R=2.

    Returns:
        depth: [H, W] float32 array in raw 16-bit units (0–65535).
    """
    r = bgr_frame[:, :, 2].astype(np.uint32)  # high byte
    g = bgr_frame[:, :, 1].astype(np.uint32)  # low byte
    depth_16 = (r * 256 + g).astype(np.float32)
    return depth_16


def depth_to_game_units(depth_16: np.ndarray, max_game_units: float = 2048.0) -> np.ndarray:
    """Convert raw 16-bit depth to CSGO game units."""
    return depth_16 / 65535.0 * max_game_units


def load_depth_frame(depth_mkv_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load a specific frame from a depth .mkv file and decode to game units.

    Returns:
        depth: [H, W] float32 in game units, or None on failure.
    """
    cap = cv2.VideoCapture(depth_mkv_path)
    if not cap.isOpened():
        logging.warning(f"Cannot open depth video: {depth_mkv_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()

    if not ret:
        logging.warning(f"Cannot read frame {frame_idx} from {depth_mkv_path}")
        return None

    depth_16 = decode_depth_frame(bgr)
    return depth_to_game_units(depth_16)


def load_video_frame(video_path: str, frame_idx: int,
                     height: int = 480, width: int = 832) -> Optional[np.ndarray]:
    """Load a specific frame from a video and resize to target resolution."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


# ============================================================
# Camera utilities
# ============================================================

def load_intrinsics(intrinsics_path: str, frame_idx: int = 0) -> np.ndarray:
    """
    Load camera intrinsics matrix [3, 3].
    intrinsics.npy may be [F, 3, 3] or [F, 4] (fx, fy, cx, cy).
    """
    K_data = np.load(intrinsics_path)
    if K_data.ndim == 3:
        K = K_data[frame_idx]   # [3, 3]
    elif K_data.ndim == 2 and K_data.shape[1] == 4:
        # [F, 4] format: fx, fy, cx, cy
        fx, fy, cx, cy = K_data[frame_idx]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        raise ValueError(f"Unexpected intrinsics shape: {K_data.shape}")
    return K.astype(np.float32)


def load_pose(poses_path: str, frame_idx: int = 0) -> np.ndarray:
    """Load camera-to-world matrix [4, 4]."""
    poses = np.load(poses_path)
    return poses[frame_idx].astype(np.float32)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Convert camera-to-world to world-to-camera."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_inv
    w2c[:3, 3] = t_inv
    return w2c


def compute_fundamental_matrix(K1: np.ndarray, c2w1: np.ndarray,
                                K2: np.ndarray, c2w2: np.ndarray) -> np.ndarray:
    """Compute fundamental matrix F from camera parameters."""
    w2c1 = c2w_to_w2c(c2w1)
    w2c2 = c2w_to_w2c(c2w2)

    # Relative pose: 2 relative to 1
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Essential matrix E = [t]x R
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0],
    ], dtype=np.float32)
    E = tx @ R_rel

    # Fundamental matrix F = K2^{-T} E K1^{-1}
    K1_inv = np.linalg.inv(K1)
    K2_inv_T = np.linalg.inv(K2).T
    F = K2_inv_T @ E @ K1_inv
    return F


def symmetric_epipolar_distance(pts1: np.ndarray, pts2: np.ndarray,
                                  F: np.ndarray) -> float:
    """
    Compute mean symmetric epipolar distance for matched point pairs.

    d_sym(p1, p2) = (p2^T F p1)^2 * (1/(||Fp1||^2) + 1/(||F^T p2||^2))
    """
    n = pts1.shape[0]
    if n == 0:
        return float("nan")

    # Homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float32)
    p1h = np.hstack([pts1, ones])  # [N, 3]
    p2h = np.hstack([pts2, ones])  # [N, 3]

    Fp1 = (F @ p1h.T).T       # [N, 3]
    FTp2 = (F.T @ p2h.T).T    # [N, 3]

    num = np.sum(p2h * Fp1, axis=1) ** 2   # [N]
    denom = (Fp1[:, 0]**2 + Fp1[:, 1]**2 + 1e-8) + \
            (FTp2[:, 0]**2 + FTp2[:, 1]**2 + 1e-8)  # [N]

    distances = num / denom
    return float(np.mean(np.abs(distances)))


def reproject_pixels(
    pts_uv: np.ndarray,
    depth_vals: np.ndarray,
    K1: np.ndarray, c2w1: np.ndarray,
    K2: np.ndarray, c2w2: np.ndarray,
    depth_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproject pixels from camera 1 to camera 2 using depth.

    Args:
        pts_uv: [N, 2] pixel coordinates in camera 1 (u, v).
        depth_vals: [N] depth values in game units for each pixel.
        K1, c2w1: Intrinsics and c2w for camera 1.
        K2, c2w2: Intrinsics and c2w for camera 2.
        depth_scale: Multiply depth by this factor (for unit conversion).

    Returns:
        pts_uv2: [N, 2] reprojected pixel coordinates in camera 2.
        valid_mask: [N] boolean mask (True if reprojected point is in image).
    """
    N = pts_uv.shape[0]
    depth = depth_vals * depth_scale

    # Back-project to 3D in camera 1 space
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    u, v = pts_uv[:, 0], pts_uv[:, 1]
    x_cam1 = (u - cx1) / fx1 * depth
    y_cam1 = (v - cy1) / fy1 * depth
    z_cam1 = depth

    # Points in camera 1 space → world space
    pts_cam1 = np.stack([x_cam1, y_cam1, z_cam1, np.ones(N)], axis=1)  # [N, 4]
    pts_world = (c2w1 @ pts_cam1.T).T  # [N, 4]

    # World → camera 2 space
    w2c2 = c2w_to_w2c(c2w2)
    pts_cam2 = (w2c2 @ pts_world.T).T  # [N, 4]

    # Project to image 2
    x2, y2, z2 = pts_cam2[:, 0], pts_cam2[:, 1], pts_cam2[:, 2]
    valid = z2 > 0.01  # in front of camera

    fx2, fy2, cx2, cy2 = K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]
    u2 = np.where(valid, fx2 * x2 / (z2 + 1e-8) + cx2, 0)
    v2 = np.where(valid, fy2 * y2 / (z2 + 1e-8) + cy2, 0)

    pts_uv2 = np.stack([u2, v2], axis=1)
    return pts_uv2, valid


# ============================================================
# Per-view-pair metrics
# ============================================================

def compute_epipolar_error_from_images(
    img1: np.ndarray, img2: np.ndarray,
    K1: np.ndarray, c2w1: np.ndarray,
    K2: np.ndarray, c2w2: np.ndarray,
    max_keypoints: int = 500,
) -> Dict:
    """
    Detect ORB keypoints in both images, match them, filter with RANSAC,
    and compute symmetric epipolar distance using known camera geometry.
    """
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return {"epipolar_error": float("nan"), "num_matches": 0, "inlier_ratio": 0.0}

    # BFMatcher with Hamming distance for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    if len(good) < 8:
        return {"epipolar_error": float("nan"), "num_matches": len(good), "inlier_ratio": 0.0}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # RANSAC using OpenCV fundamental matrix to find inliers
    _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    if mask is None:
        inlier_ratio = 0.0
        inlier_pts1, inlier_pts2 = pts1, pts2
    else:
        mask = mask.ravel().astype(bool)
        inlier_ratio = float(mask.sum()) / len(mask) if len(mask) > 0 else 0.0
        inlier_pts1 = pts1[mask]
        inlier_pts2 = pts2[mask]

    # Compute epipolar error using GT camera geometry
    F_gt = compute_fundamental_matrix(K1, c2w1, K2, c2w2)
    epi_err = symmetric_epipolar_distance(inlier_pts1, inlier_pts2, F_gt)

    return {
        "epipolar_error": epi_err,
        "num_matches": len(good),
        "inlier_ratio": inlier_ratio,
        "num_inliers": int(mask.sum()) if mask is not None else len(good),
    }


def compute_reprojection_psnr(
    gen_frame1: np.ndarray, gen_frame2: np.ndarray,
    depth_frame1: np.ndarray,
    K1: np.ndarray, c2w1: np.ndarray,
    K2: np.ndarray, c2w2: np.ndarray,
    H2: int, W2: int,
    sample_rate: float = 0.05,
) -> float:
    """
    Reproject pixels from generated view 1 to view 2 using GT depth,
    compare reprojected colors to generated view 2.

    This measures: does the appearance generated by player A match
    what player B sees at the geometrically corresponding location?
    """
    H1, W1 = depth_frame1.shape

    # Resize gen_frame1 to depth resolution
    gen1_resized = cv2.resize(gen_frame1, (W1, H1), interpolation=cv2.INTER_LINEAR)
    gen2_resized = cv2.resize(gen_frame2, (W2, H2), interpolation=cv2.INTER_LINEAR)

    # Sample pixels from view 1
    n_pixels = int(H1 * W1 * sample_rate)
    ys = np.random.randint(0, H1, n_pixels)
    xs = np.random.randint(0, W1, n_pixels)
    pts1 = np.stack([xs, ys], axis=1).astype(np.float32)
    depths1 = depth_frame1[ys, xs]

    # Filter out invalid (zero) depth
    valid_depth = depths1 > 0.5
    pts1 = pts1[valid_depth]
    depths1 = depths1[valid_depth]
    ys, xs = ys[valid_depth], xs[valid_depth]

    if len(pts1) == 0:
        return float("nan")

    # Scale intrinsics for depth resolution
    scale_x = W1 / W2
    scale_y = H1 / H2
    K1_scaled = K1.copy()
    K1_scaled[0] *= scale_x
    K1_scaled[1] *= scale_y
    K2_scaled = K2.copy()
    K2_scaled[0] *= scale_x
    K2_scaled[1] *= scale_y

    # Reproject to view 2
    pts2, valid = reproject_pixels(pts1, depths1, K1_scaled, c2w1, K2_scaled, c2w2)

    # Clip to image bounds
    valid &= (pts2[:, 0] >= 0) & (pts2[:, 0] < W1 - 1)
    valid &= (pts2[:, 1] >= 0) & (pts2[:, 1] < H1 - 1)

    if valid.sum() == 0:
        return float("nan")

    # Sample colors from both views
    xs_src = xs[valid]
    ys_src = ys[valid]
    u2 = pts2[valid, 0].astype(np.int32)
    v2 = pts2[valid, 1].astype(np.int32)

    colors1 = gen1_resized[ys_src, xs_src].astype(np.float32)
    colors2 = gen2_resized[v2, u2].astype(np.float32)

    mse = np.mean((colors1 - colors2) ** 2)
    if mse < 1e-10:
        return 100.0
    psnr = 10 * math.log10(255.0 ** 2 / mse)
    return float(psnr)


# ============================================================
# Episode / clip discovery
# ============================================================

def load_metadata(dataset_dir: str, split: str) -> List[Dict]:
    """Load clip metadata from CSV."""
    csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
    clips = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_dir = os.path.join(dataset_dir, row["clip_path"])
            clips.append({
                "clip_dir": clip_dir,
                "episode_id": row["episode_id"],
                "stem": row["stem"],
            })
    return clips


def group_by_episode(clips: List[Dict]) -> Dict[str, List[Dict]]:
    """Group clips by episode_id."""
    episodes = {}
    for c in clips:
        ep = c["episode_id"]
        episodes.setdefault(ep, []).append(c)
    return episodes


def find_depth_video(raw_data_dir: str, stem: str) -> Optional[str]:
    """Find the depth .mkv file for a given stem."""
    # Raw data structure: Ep_XXXXXX/<stem>_depth.mkv
    episode_id = stem.split("_")[1] if "_" in stem else stem
    ep_dir = os.path.join(raw_data_dir, f"Ep_{episode_id}")
    depth_path = os.path.join(ep_dir, f"{stem}_depth.mkv")
    if os.path.exists(depth_path):
        return depth_path
    # Fallback: search
    for root, _, files in os.walk(raw_data_dir):
        for f in files:
            if f == f"{stem}_depth.mkv":
                return os.path.join(root, f)
    return None


# ============================================================
# Main evaluation
# ============================================================

def evaluate_episode_pair(
    clip_a: Dict, clip_b: Dict,
    gen_dir_stage2: Optional[str],
    gen_dir_stage1: Optional[str],
    raw_data_dir: str,
    frame_idx: int = 20,
    H: int = 480, W: int = 832,
) -> Dict:
    """
    Evaluate cross-view consistency for a pair of clips from the same episode.
    """
    result = {
        "episode_id": clip_a["episode_id"],
        "stem_a": clip_a["stem"],
        "stem_b": clip_b["stem"],
    }

    # ---- Load camera data ----
    try:
        K_a = load_intrinsics(os.path.join(clip_a["clip_dir"], "intrinsics.npy"), frame_idx)
        K_b = load_intrinsics(os.path.join(clip_b["clip_dir"], "intrinsics.npy"), frame_idx)
        c2w_a = load_pose(os.path.join(clip_a["clip_dir"], "poses.npy"), frame_idx)
        c2w_b = load_pose(os.path.join(clip_b["clip_dir"], "poses.npy"), frame_idx)
    except Exception as e:
        logging.warning(f"  Cannot load camera data: {e}")
        return result

    # ---- Load GT depth for view A ----
    depth_a = None
    depth_path_a = find_depth_video(raw_data_dir, clip_a["stem"])
    if depth_path_a:
        depth_a = load_depth_frame(depth_path_a, frame_idx)
    else:
        logging.debug(f"  No depth video found for {clip_a['stem']}")

    # ---- Helper to get generated frame ----
    def get_gen_frame(clip, gen_dir):
        if gen_dir is None:
            return None
        clip_name = os.path.basename(clip["clip_dir"])
        gen_path = os.path.join(gen_dir, f"{clip_name}_gen.mp4")
        if not os.path.exists(gen_path):
            # Also try by stem
            gen_path = os.path.join(gen_dir, f"{clip['stem']}.mp4")
        if not os.path.exists(gen_path):
            return None
        return load_video_frame(gen_path, frame_idx, H, W)

    # ---- Stage 2 metrics ----
    gen_a_s2 = get_gen_frame(clip_a, gen_dir_stage2)
    gen_b_s2 = get_gen_frame(clip_b, gen_dir_stage2)
    if gen_a_s2 is not None and gen_b_s2 is not None:
        epi = compute_epipolar_error_from_images(
            gen_a_s2, gen_b_s2, K_a, c2w_a, K_b, c2w_b
        )
        result.update({
            "s2_epipolar_error": epi["epipolar_error"],
            "s2_num_matches": epi["num_matches"],
            "s2_inlier_ratio": epi["inlier_ratio"],
        })
        if depth_a is not None:
            reproj_psnr = compute_reprojection_psnr(
                gen_a_s2, gen_b_s2, depth_a, K_a, c2w_a, K_b, c2w_b, H, W
            )
            result["s2_reproj_psnr"] = reproj_psnr

    # ---- Stage 1 (baseline) metrics ----
    gen_a_s1 = get_gen_frame(clip_a, gen_dir_stage1)
    gen_b_s1 = get_gen_frame(clip_b, gen_dir_stage1)
    if gen_a_s1 is not None and gen_b_s1 is not None:
        epi = compute_epipolar_error_from_images(
            gen_a_s1, gen_b_s1, K_a, c2w_a, K_b, c2w_b
        )
        result.update({
            "s1_epipolar_error": epi["epipolar_error"],
            "s1_num_matches": epi["num_matches"],
            "s1_inlier_ratio": epi["inlier_ratio"],
        })
        if depth_a is not None:
            reproj_psnr = compute_reprojection_psnr(
                gen_a_s1, gen_b_s1, depth_a, K_a, c2w_a, K_b, c2w_b, H, W
            )
            result["s1_reproj_psnr"] = reproj_psnr

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-View Consistency Evaluation")
    parser.add_argument("--stage2_gen_dir", type=str, default="",
                        help="Directory with Stage 2 generated videos (*_gen.mp4)")
    parser.add_argument("--stage1_gen_dir", type=str, default="",
                        help="Directory with Stage 1 generated videos (baseline)")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Processed dataset dir (contains metadata_val.csv)")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                        help="Raw data dir (contains Ep_XXXXXX/*_depth.mkv)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_episode_pairs", type=int, default=50,
                        help="Max cross-view pairs to evaluate (0 = all)")
    parser.add_argument("--frame_idx", type=int, default=20,
                        help="Which frame to use for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load and group clips ----
    logging.info(f"Loading {args.split} metadata from {args.dataset_dir}")
    clips = load_metadata(args.dataset_dir, args.split)
    episodes = group_by_episode(clips)
    logging.info(f"Found {len(clips)} clips across {len(episodes)} episodes")

    # ---- Build evaluation pairs ----
    pairs = []
    for ep_id, ep_clips in episodes.items():
        if len(ep_clips) < 2:
            continue
        # Take pairs from the same episode (max 3 pairs per episode)
        random.shuffle(ep_clips)
        for i in range(0, min(len(ep_clips) - 1, 6), 2):
            pairs.append((ep_clips[i], ep_clips[i + 1]))

    if args.max_episode_pairs > 0:
        random.shuffle(pairs)
        pairs = pairs[:args.max_episode_pairs]

    logging.info(f"Evaluating {len(pairs)} cross-view pairs")

    # ---- Evaluate ----
    results = []
    for idx, (clip_a, clip_b) in enumerate(pairs):
        logging.info(
            f"[{idx+1}/{len(pairs)}] "
            f"Ep {clip_a['episode_id']}: "
            f"{os.path.basename(clip_a['clip_dir'])} ↔ "
            f"{os.path.basename(clip_b['clip_dir'])}"
        )
        try:
            r = evaluate_episode_pair(
                clip_a, clip_b,
                gen_dir_stage2=args.stage2_gen_dir or None,
                gen_dir_stage1=args.stage1_gen_dir or None,
                raw_data_dir=args.raw_data_dir,
                frame_idx=args.frame_idx,
            )
            results.append(r)

            parts = []
            for k in ["s2_epipolar_error", "s2_reproj_psnr", "s1_epipolar_error"]:
                if k in r and not (isinstance(r[k], float) and math.isnan(r[k])):
                    parts.append(f"{k}={r[k]:.3f}")
            if parts:
                logging.info(f"  " + ", ".join(parts))
        except Exception as e:
            logging.error(f"  Error: {e}")
            results.append({
                "episode_id": clip_a["episode_id"],
                "stem_a": clip_a["stem"],
                "stem_b": clip_b["stem"],
                "error": str(e),
            })

    # ---- Aggregate ----
    valid = [r for r in results if "error" not in r]
    report = {"num_pairs": len(pairs), "num_valid": len(valid), "aggregate": {}}

    metrics = ["s2_epipolar_error", "s2_reproj_psnr", "s2_inlier_ratio",
               "s1_epipolar_error", "s1_reproj_psnr", "s1_inlier_ratio"]
    for m in metrics:
        vals = [r[m] for r in valid if m in r
                and isinstance(r[m], float) and not math.isnan(r[m])]
        if vals:
            report["aggregate"][m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
                "n":    len(vals),
            }

    report["per_pair"] = results

    # ---- Print summary ----
    logging.info("\n" + "=" * 60)
    logging.info("CROSS-VIEW CONSISTENCY SUMMARY")
    logging.info("=" * 60)
    agg = report["aggregate"]
    row_fmt = "{:<30} {:>10} {:>10}"
    logging.info(row_fmt.format("Metric", "Stage2(ours)", "Stage1(base)"))
    logging.info("-" * 52)
    for s2k, s1k, label in [
        ("s2_epipolar_error", "s1_epipolar_error", "Epipolar Error (↓)"),
        ("s2_reproj_psnr",    "s1_reproj_psnr",    "Reproj PSNR (↑)"),
        ("s2_inlier_ratio",   "s1_inlier_ratio",   "Match Inlier Rate (↑)"),
    ]:
        s2 = f"{agg[s2k]['mean']:.4f}" if s2k in agg else "  —"
        s1 = f"{agg[s1k]['mean']:.4f}" if s1k in agg else "  —"
        logging.info(row_fmt.format(label, s2, s1))
    logging.info("=" * 60)

    # ---- Save ----
    report_path = os.path.join(args.output_dir, "crossview_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved: {report_path}")

    csv_path = os.path.join(args.output_dir, "crossview_pairs.csv")
    if valid:
        fieldnames = list(dict.fromkeys(k for r in results for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
    logging.info(f"Per-pair CSV: {csv_path}")


if __name__ == "__main__":
    main()
