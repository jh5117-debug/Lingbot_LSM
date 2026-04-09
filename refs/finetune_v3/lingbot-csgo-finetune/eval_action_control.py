"""
Action Controllability Evaluation for CSGO Fine-tuned LingBot-World.

Measures whether generated videos actually follow the input actions.
Three metrics:
  1. Optical Flow Direction Accuracy — cosine similarity between expected
     camera motion direction and dominant optical flow direction
  2. Trajectory Consistency — correlation between optical flow magnitude
     and pose displacement (moving when told to move, still when told to stop)
  3. Turn Direction Accuracy — sign agreement between yaw change and
     horizontal flow direction (left turn -> rightward flow, etc.)

Compares generated videos (*_gen.mp4 from eval_batch.py) against
action/pose data (poses.npy, action.npy in clip directories).

Usage:
    python eval_action_control.py \
        --gen_dir   /path/to/eval_output/videos \
        --clip_dir  /path/to/processed_csgo_v3/val/clips \
        --output_dir /path/to/action_control_output

    # Evaluate first 10 clips only:
    python eval_action_control.py \
        --gen_dir ... --clip_dir ... --output_dir ... --max_samples 10
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# Video I/O helpers
# ===========================================================================

def read_video_frames(path: str, max_frames: int = 81) -> np.ndarray:
    """Read video -> uint8 np.ndarray [T, H, W, 3] BGR (for optical flow)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def compute_optical_flow(frames: np.ndarray):
    """
    Compute dense optical flow (Farneback) between consecutive frames.

    Args:
        frames: [T, H, W, 3] uint8 BGR

    Returns:
        flows: list of [H, W, 2] float32 arrays (dx, dy), length T-1
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flows = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow)
    return flows


# ===========================================================================
# Pose / Action utilities
# ===========================================================================

def extract_yaw_from_poses(poses: np.ndarray) -> np.ndarray:
    """
    Extract yaw angle (rotation around Y-axis) from c2w matrices.

    Args:
        poses: [T, 4, 4] camera-to-world matrices

    Returns:
        yaw: [T] array of yaw angles in radians
    """
    # Forward direction is -Z column of rotation matrix
    forward = -poses[:, :3, 2]  # [T, 3]
    # Yaw = atan2(forward_x, forward_z) in world space (XZ plane)
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    return yaw


def compute_expected_motion_2d(poses: np.ndarray):
    """
    Compute expected 2D screen-space motion direction from camera poses.

    For a forward-facing camera:
      - Forward motion -> expansion from center (approximated as upward flow
        in screen coords since ground plane dominates FPS view)
      - Left/right strafe -> horizontal flow
      - Yaw rotation -> horizontal flow

    We project the 3D displacement into the camera's local frame to get
    expected screen-space flow direction.

    Args:
        poses: [T, 4, 4] c2w matrices

    Returns:
        expected_flow_dir: [T-1, 2] normalized (dx, dy) expected flow direction
            in image coordinates (x=right, y=down)
        displacements: [T-1] magnitude of 3D displacement
    """
    positions = poses[:, :3, 3]  # [T, 3]
    displacements_3d = positions[1:] - positions[:-1]  # [T-1, 3]
    magnitudes = np.linalg.norm(displacements_3d, axis=1)  # [T-1]

    expected = np.zeros((len(displacements_3d), 2), dtype=np.float32)

    for t in range(len(displacements_3d)):
        # Get camera axes at frame t
        R = poses[t, :3, :3]  # rotation part of c2w
        cam_right = R[:, 0]    # X column = right
        cam_up = R[:, 1]       # Y column = up (but screen y is down)
        cam_forward = -R[:, 2]  # -Z column = forward (looking direction)

        disp = displacements_3d[t]

        # Project displacement onto camera-local axes
        dx_local = np.dot(disp, cam_right)    # rightward motion
        dy_local = -np.dot(disp, cam_up)      # downward in screen (flip Y)
        dz_local = np.dot(disp, cam_forward)  # forward motion

        # Forward motion causes radial expansion from center.
        # In practice for FPS games, forward motion appears as downward flow
        # in the lower half and upward in the upper half. We approximate by
        # noting that forward motion causes *negative* mean vertical flow
        # (scene rushes toward you = things move outward from center, but
        # ground dominates -> net downward flow in screen).
        # We encode forward motion as a downward flow component.
        screen_dx = -dx_local  # camera moves right -> scene flows left
        screen_dy = -dy_local + dz_local * 0.3  # forward adds some downward flow

        norm = np.sqrt(screen_dx ** 2 + screen_dy ** 2)
        if norm > 1e-8:
            expected[t] = [screen_dx / norm, screen_dy / norm]

    return expected, magnitudes


# ===========================================================================
# Metric 1: Optical Flow Direction Accuracy
# ===========================================================================

def metric_flow_direction_accuracy(flows, expected_flow_dir, displacements,
                                   min_displacement: float = 0.001):
    """
    Cosine similarity between expected motion direction and actual
    dominant optical flow direction.

    Only evaluated on frames where displacement > min_displacement.

    Returns:
        mean cosine similarity, per-frame scores
    """
    n = min(len(flows), len(expected_flow_dir))
    scores = []

    for t in range(n):
        if displacements[t] < min_displacement:
            continue  # skip near-stationary frames

        flow = flows[t]  # [H, W, 2]
        # Compute mean flow direction
        mean_dx = np.mean(flow[:, :, 0])
        mean_dy = np.mean(flow[:, :, 1])
        flow_norm = np.sqrt(mean_dx ** 2 + mean_dy ** 2)

        if flow_norm < 1e-8:
            scores.append(0.0)
            continue

        actual_dir = np.array([mean_dx / flow_norm, mean_dy / flow_norm])
        expected_dir = expected_flow_dir[t]

        if np.linalg.norm(expected_dir) < 1e-8:
            continue

        cos_sim = float(np.dot(actual_dir, expected_dir))
        scores.append(cos_sim)

    if not scores:
        return 0.0, []

    return float(np.mean(scores)), scores


# ===========================================================================
# Metric 2: Trajectory Consistency
# ===========================================================================

def metric_trajectory_consistency(flows, displacements, actions):
    """
    Correlation between optical flow magnitude and pose displacement.

    Also computes:
      - stop_violation: fraction of "stop" frames with large flow
      - move_violation: fraction of "move" frames with near-zero flow

    Args:
        flows: list of [H, W, 2] optical flow arrays
        displacements: [T-1] 3D displacement magnitudes
        actions: [T, 4] WASD binary actions (forward/back/left/right)

    Returns:
        correlation, stop_violation_rate, move_violation_rate
    """
    n = min(len(flows), len(displacements))
    if n < 2:
        return 0.0, 0.0, 0.0

    flow_mags = np.array([
        np.mean(np.sqrt(f[:, :, 0] ** 2 + f[:, :, 1] ** 2))
        for f in flows[:n]
    ])

    disp = displacements[:n]

    # Pearson correlation
    if np.std(flow_mags) < 1e-8 or np.std(disp) < 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flow_mags, disp)[0, 1])

    # Stop violation: action says stop (all WASD=0), but flow is large
    # Move violation: action says move (any WASD=1), but flow is tiny
    flow_median = np.median(flow_mags) if len(flow_mags) > 0 else 1.0
    stop_threshold = flow_median * 1.5  # flow above this during "stop" = violation
    move_threshold = flow_median * 0.3  # flow below this during "move" = violation

    stop_violations = 0
    stop_total = 0
    move_violations = 0
    move_total = 0

    for t in range(n):
        if t >= len(actions):
            break
        is_moving = np.any(actions[t] > 0)

        if not is_moving:
            stop_total += 1
            if flow_mags[t] > stop_threshold:
                stop_violations += 1
        else:
            move_total += 1
            if flow_mags[t] < move_threshold:
                move_violations += 1

    stop_violation_rate = stop_violations / max(stop_total, 1)
    move_violation_rate = move_violations / max(move_total, 1)

    return correlation, stop_violation_rate, move_violation_rate


# ===========================================================================
# Metric 3: Turn Direction Accuracy
# ===========================================================================

def metric_turn_direction_accuracy(flows, poses, min_yaw_change: float = 0.005):
    """
    When yaw changes, check if optical flow shows corresponding rotation.
    Left turn (positive yaw change) -> rightward flow (positive mean dx)
    Right turn (negative yaw change) -> leftward flow (negative mean dx)

    Args:
        flows: list of [H, W, 2] optical flow arrays
        poses: [T, 4, 4] c2w matrices
        min_yaw_change: minimum yaw delta to count as a turn (radians)

    Returns:
        sign_agreement_ratio, num_turning_frames
    """
    yaw = extract_yaw_from_poses(poses)
    yaw_deltas = yaw[1:] - yaw[:-1]

    # Unwrap large jumps (e.g., -pi to pi transition)
    yaw_deltas = np.where(yaw_deltas > np.pi, yaw_deltas - 2 * np.pi, yaw_deltas)
    yaw_deltas = np.where(yaw_deltas < -np.pi, yaw_deltas + 2 * np.pi, yaw_deltas)

    n = min(len(flows), len(yaw_deltas))
    agreements = 0
    total = 0

    for t in range(n):
        if abs(yaw_deltas[t]) < min_yaw_change:
            continue  # not a turning frame

        mean_dx = np.mean(flows[t][:, :, 0])

        # Left turn (positive yaw delta in world) should cause rightward
        # optical flow (scene moves right as camera turns left)
        expected_sign = np.sign(yaw_deltas[t])
        actual_sign = np.sign(mean_dx)

        if expected_sign == actual_sign:
            agreements += 1
        total += 1

    if total == 0:
        return 1.0, 0  # no turning frames -> vacuously correct

    return float(agreements / total), total


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


def match_gen_to_clips(gen_paths, clip_dir: str):
    """
    Match generated videos to clip directories.
    gen name pattern: <clip_stem>_gen.mp4
    clip dir pattern: clip_dir/<clip_stem>/

    Returns list of (gen_path, clip_path) tuples.
    """
    clip_dir = Path(clip_dir)

    # Build lookup: clip_stem -> clip_path
    clip_lookup = {}
    for d in sorted(clip_dir.iterdir()):
        if d.is_dir():
            clip_lookup[d.name] = str(d)

    pairs = []
    for gp in gen_paths:
        gname = Path(gp).stem  # e.g. "Ep_000005_team_2_player_0000_clip0020_gen"
        # Remove _gen suffix
        if gname.endswith("_gen"):
            clip_stem = gname[:-4]
        else:
            clip_stem = gname

        if clip_stem in clip_lookup:
            pairs.append((gp, clip_lookup[clip_stem]))
        else:
            log.debug(f"No matching clip for {gname}")

    return pairs


# ===========================================================================
# Per-clip evaluation
# ===========================================================================

def evaluate_clip(gen_video_path: str, clip_path: str) -> dict:
    """
    Evaluate action controllability for a single generated video.

    Args:
        gen_video_path: path to generated video (*_gen.mp4)
        clip_path: path to clip directory (with poses.npy, action.npy)

    Returns:
        dict with metric scores
    """
    clip_name = os.path.basename(clip_path)
    result = {"clip_name": clip_name}

    # Load data
    poses_path = os.path.join(clip_path, "poses.npy")
    action_path = os.path.join(clip_path, "action.npy")

    if not os.path.exists(poses_path):
        log.warning(f"  poses.npy not found in {clip_path}")
        result["error"] = "poses.npy missing"
        return result

    if not os.path.exists(action_path):
        log.warning(f"  action.npy not found in {clip_path}")
        result["error"] = "action.npy missing"
        return result

    poses = np.load(poses_path)   # [81, 4, 4]
    actions = np.load(action_path)  # [81, 4]

    # Read generated video
    frames = read_video_frames(gen_video_path, max_frames=81)
    if len(frames) < 2:
        log.warning(f"  Generated video too short: {len(frames)} frames")
        result["error"] = f"video too short ({len(frames)} frames)"
        return result

    result["num_frames"] = len(frames)

    # Compute optical flow
    flows = compute_optical_flow(frames)

    # --- Metric 1: Flow Direction Accuracy ---
    expected_flow_dir, displacements = compute_expected_motion_2d(poses)
    flow_dir_acc, flow_dir_scores = metric_flow_direction_accuracy(
        flows, expected_flow_dir, displacements
    )
    result["flow_direction_accuracy"] = round(flow_dir_acc, 4)
    result["flow_direction_num_frames"] = len(flow_dir_scores)

    # --- Metric 2: Trajectory Consistency ---
    correlation, stop_viol, move_viol = metric_trajectory_consistency(
        flows, displacements, actions
    )
    result["trajectory_correlation"] = round(correlation, 4)
    result["stop_violation_rate"] = round(stop_viol, 4)
    result["move_violation_rate"] = round(move_viol, 4)

    # --- Metric 3: Turn Direction Accuracy ---
    turn_acc, num_turning = metric_turn_direction_accuracy(flows, poses)
    result["turn_direction_accuracy"] = round(turn_acc, 4)
    result["turn_num_frames"] = num_turning

    return result


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Action controllability evaluation for CSGO video generation"
    )
    p.add_argument("--gen_dir", required=True,
                   help="Directory containing generated videos (*_gen.mp4)")
    p.add_argument("--clip_dir", required=True,
                   help="Processed clip directory (each clip has poses.npy, action.npy)")
    p.add_argument("--output_dir", required=True,
                   help="Where to save eval_action_control_report.json")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Max clips to evaluate (0 = all)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Discover videos ----
    gen_paths = find_gen_videos(args.gen_dir)
    log.info(f"Found {len(gen_paths)} generated videos in {args.gen_dir}")

    pairs = match_gen_to_clips(gen_paths, args.clip_dir)
    log.info(f"Matched {len(pairs)} gen-clip pairs")

    if not pairs:
        log.error("No matching pairs found. Check --gen_dir and --clip_dir.")
        return

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    n = len(pairs)
    log.info(f"Evaluating {n} clips")

    # ---- Per-clip evaluation ----
    try:
        from tqdm import tqdm
        iterator = tqdm(pairs, desc="Evaluating action control")
    except ImportError:
        iterator = pairs

    results = []
    t0 = time.time()

    for gen_path, clip_path in iterator:
        clip_name = os.path.basename(clip_path)
        log.info(f"Processing {clip_name}")

        try:
            result = evaluate_clip(gen_path, clip_path)
            results.append(result)

            if "error" not in result:
                log.info(
                    f"  FlowDirAcc={result['flow_direction_accuracy']:.4f}  "
                    f"TrajCorr={result['trajectory_correlation']:.4f}  "
                    f"TurnAcc={result['turn_direction_accuracy']:.4f}"
                )
        except Exception as e:
            log.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"clip_name": clip_name, "error": str(e)})

    elapsed = time.time() - t0

    # ---- Aggregate metrics ----
    valid = [r for r in results if "error" not in r]

    aggregate = {}
    metric_keys = [
        "flow_direction_accuracy",
        "trajectory_correlation",
        "stop_violation_rate",
        "move_violation_rate",
        "turn_direction_accuracy",
    ]

    for key in metric_keys:
        values = [r[key] for r in valid if key in r and np.isfinite(r[key])]
        if values:
            aggregate[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }

    # ---- Build report ----
    report = {
        "config": {
            "gen_dir": args.gen_dir,
            "clip_dir": args.clip_dir,
            "num_clips": n,
            "num_evaluated": len(valid),
            "num_errors": len(results) - len(valid),
            "eval_time_s": round(elapsed, 1),
        },
        "aggregate_metrics": aggregate,
        "per_clip": results,
    }

    # ---- Save report ----
    out_path = os.path.join(args.output_dir, "eval_action_control_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved -> {out_path}")

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("ACTION CONTROLLABILITY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Clips evaluated  : {len(valid)} / {n}")
    print(f"  Eval time        : {elapsed:.1f}s")
    print("-" * 60)
    if aggregate:
        for key, stats in aggregate.items():
            label = key.replace("_", " ").title()
            print(f"  {label:30s}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
                  f"(min={stats['min']:.4f}, max={stats['max']:.4f})")
    else:
        print("  No valid results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
