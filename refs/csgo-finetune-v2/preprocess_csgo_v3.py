"""
CSGO Dataset Preprocessor v3 for LingBot-World Fine-tuning
===========================================================
Adapted for the new Solaris dataset format (2026-03-25+):
  - Per-episode subdirectories: train/Ep_NNNNNN/<stem>.*
  - 64 FPS video (tf_ratio=2, tickrate=128)
  - frame_count in JSON jumps by tf_ratio (0, 2, 4, ...)
  - game_manifest.json per episode (map_name, player info)
  - episode_info has explicit tf_ratio, video_fps, start_tick
  - Richer action fields (jump, crouch, walk, fire, reload, use, look_dx, look_dy, weapon_slot)
  - camera_rotation vec3 available (replaces root yaw/pitch for rotation)
  - 10 players per episode (5v5), each with independent video + action + pose

Usage:
    python preprocess_csgo_v3.py \\
        --input_dir /home/nvme02/lingbot-world/datasets/new_sample/bcc3a2ac490c4888bcb97c3553a25671/ \\
        --output_dir /home/nvme02/lingbot-world/datasets/csgo_processed_v3/ \\
        --clip_frames 81 \\
        --target_fps 16 \\
        --height 480 --width 832 \\
        --stride 40 \\
        --skip_episodes "000004"
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Discovery
# ============================================================

def find_player_streams(input_dir, skip_episodes=None):
    """
    Walk the new dataset structure and find all per-player streams.
    
    New structure:
        input_dir/
          train/
            Ep_000005/
              Ep_000005_team_2_player_0000_inst_000.json
              Ep_000005_team_2_player_0000_inst_000.mp4
              Ep_000005_team_2_player_0000_inst_000_episode_info.json
              game_manifest.json
              ...
          test/
            ...

    Returns list of dicts, one per player-stream.
    """
    skip_set = set(skip_episodes or [])
    streams = []

    for split_name in ["train", "test"]:
        split_dir = os.path.join(input_dir, split_name)
        if not os.path.isdir(split_dir):
            continue

        for ep_dir_name in sorted(os.listdir(split_dir)):
            ep_dir = os.path.join(split_dir, ep_dir_name)
            if not os.path.isdir(ep_dir):
                continue

            # Extract episode id from dir name (e.g. "Ep_000005" -> "000005")
            if not ep_dir_name.startswith("Ep_"):
                continue
            episode_id = ep_dir_name.replace("Ep_", "")

            if episode_id in skip_set:
                print(f"[SKIP] Episode {episode_id} (in skip list)")
                continue

            # Load game manifest for map name
            manifest_path = os.path.join(ep_dir, "game_manifest.json")
            map_name = "unknown"
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                map_name = manifest.get("map_name", "unknown")

            # Find all episode_info files to discover player streams
            for fname in sorted(os.listdir(ep_dir)):
                if not fname.endswith("_episode_info.json"):
                    continue

                stem = fname.replace("_episode_info.json", "")
                info_path = os.path.join(ep_dir, fname)
                video_path = os.path.join(ep_dir, f"{stem}.mp4")
                action_path = os.path.join(ep_dir, f"{stem}.json")

                if not os.path.exists(video_path):
                    print(f"[WARN] Video not found: {video_path}, skipping")
                    continue
                if not os.path.exists(action_path):
                    print(f"[WARN] Action JSON not found: {action_path}, skipping")
                    continue

                with open(info_path, "r") as f:
                    info = json.load(f)

                # Skip streams with errors
                if info.get("encountered_error", False):
                    print(f"[WARN] {stem}: encountered_error=true, skipping")
                    continue

                streams.append({
                    "stem": stem,
                    "episode_id": episode_id,
                    "split": split_name,
                    "video_path": video_path,
                    "action_path": action_path,
                    "info": info,
                    "ep_dir": ep_dir,
                    "map_name": map_name,
                    "video_fps": info.get("video_fps", 64),
                    "tf_ratio": info.get("tf_ratio", 2),
                    "tickrate": info.get("tickrate", 128),
                })

    print(f"Found {len(streams)} player streams across {len(set(s['episode_id'] for s in streams))} episodes")
    return streams


# ============================================================
# Coordinate conversion
# ============================================================

def csgo_to_pose_matrix(yaw_deg, pitch_deg, x, y, z):
    """
    Convert CSGO yaw/pitch/position to 4x4 camera-to-world matrix (OpenCV convention).

    Source engine coordinate system:
        x = forward, y = left, z = up
    OpenCV coordinate system:
        x = right, y = down, z = forward
    
    Pitch/yaw are in degrees. Pitch range should be ±90 (looking up/down).
    Yaw range is 0-360 or ±180 (both handled).
    """
    # Normalize angles to ±180 range
    if yaw_deg > 180:
        yaw_deg -= 360
    if pitch_deg > 180:
        pitch_deg -= 360

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # Forward direction in Source coords
    forward = np.array([cp * cy, cp * sy, -sp])
    # Right direction (Source y is left, so right = -y direction)
    right = np.array([-sy, cy, 0.0])
    # Up direction
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # Convert to OpenCV convention: x=right, y=down, z=forward
    R_opencv = np.stack([right, -up, forward], axis=1)

    pose = np.eye(4)
    pose[:3, :3] = R_opencv
    pose[:3, 3] = np.array([x, y, z])
    return pose


def fov_to_intrinsics(fov_deg, height, width):
    """Convert horizontal FOV to camera intrinsics [fx, fy, cx, cy]."""
    fov_rad = np.radians(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float32)


# ============================================================
# Video extraction
# ============================================================

def extract_video_frames(video_path, video_fps, target_fps):
    """
    Extract frames from video, downsampling from video_fps to target_fps.
    Returns list of (frame_bgr, video_frame_index) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # Use actual FPS from file if available, otherwise use provided video_fps
    if actual_fps > 0 and abs(actual_fps - video_fps) < 5:
        source_fps = actual_fps
    else:
        source_fps = video_fps

    # Calculate frame skip: e.g. 64fps -> 16fps = skip every 4th frame
    skip = max(1, round(source_fps / target_fps))

    frames = []
    frame_indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            frames.append(frame)
            frame_indices.append(idx)
        idx += 1

    cap.release()
    return frames, frame_indices, source_fps


# ============================================================
# Action JSON loading & frame alignment
# ============================================================

def load_action_data(action_path):
    """Load action JSON array. Returns list of dicts + frame_count-indexed lookup."""
    with open(action_path, "r") as f:
        frames = json.load(f)

    # Build lookup by frame_count for fast access
    # New format: frame_count jumps (0, 2, 4, ...) due to tf_ratio=2
    by_frame_count = {}
    for af in frames:
        fc = af.get("frame_count", af.get("frame_idx", af.get("frame_index", None)))
        if fc is not None:
            by_frame_count[fc] = af

    return frames, by_frame_count


def align_video_to_action(video_frame_indices, by_frame_count, tf_ratio):
    """
    Map video frame indices to action frame data.
    
    Video is at video_fps (e.g. 64fps).
    Action JSON has one entry per tf_ratio video frames (e.g. every 2nd frame).
    So action frame_count = video_frame_index (since frame_count counts video frames,
    but only every tf_ratio-th frame has an action entry).
    
    For video frame i, the corresponding action frame_count = i 
    (but only even frame_counts exist in the JSON when tf_ratio=2).
    We map each video frame to the nearest available action frame.
    """
    aligned = []
    for vid_idx in video_frame_indices:
        # frame_count in the JSON corresponds directly to video frame index
        # but only multiples of tf_ratio exist
        # Find nearest: round down to nearest multiple of tf_ratio
        fc = (vid_idx // tf_ratio) * tf_ratio
        af = by_frame_count.get(fc)
        if af is None:
            # Try adjacent
            for offset in [tf_ratio, -tf_ratio, 2*tf_ratio]:
                af = by_frame_count.get(fc + offset)
                if af is not None:
                    break
        aligned.append(af)
    return aligned


# ============================================================
# Clip extraction
# ============================================================

def extract_pose_from_action_frame(af, default_fov=106.26):
    """
    Extract camera pose and action from a single action frame dict.
    Returns (pose_4x4, action_4d, fov) or None if frame is invalid.
    """
    if af is None:
        return None

    # Get camera position (eye height, more accurate than root x/y/z)
    cam_pos = af.get("camera_position")
    if cam_pos is not None:
        x, y, z = cam_pos[0], cam_pos[1], cam_pos[2]
    else:
        x = af.get("x", 0.0)
        y = af.get("y", 0.0)
        z = af.get("z", 0.0)

    # Get rotation - prefer camera_rotation if available
    cam_rot = af.get("camera_rotation")
    if cam_rot is not None:
        # camera_rotation is [pitch, roll, yaw] in degrees
        pitch = cam_rot[0]
        yaw = cam_rot[2]
    else:
        yaw = af.get("yaw", 0.0)
        pitch = af.get("pitch", 0.0)

    pose = csgo_to_pose_matrix(yaw, pitch, x, y, z)

    # Action: [forward, back, left, right] as int 0/1
    act = af.get("action", {})
    action = [
        int(bool(act.get("forward", False))),
        int(bool(act.get("back", False))),
        int(bool(act.get("left", False))),
        int(bool(act.get("right", False))),
    ]

    return pose, action, default_fov


def is_static_clip(poses, rot_threshold_deg=1.0, trans_threshold=1e-3):
    """Check if a clip has almost no camera movement."""
    translations = poses[:, :3, 3]
    trans_range = translations.max(axis=0) - translations.min(axis=0)

    rotations = poses[:, :3, :3]
    rot_changes = []
    for i in range(1, len(rotations)):
        rel_rot = rotations[i] @ rotations[i-1].T
        trace = np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)
        rot_changes.append(abs(np.arccos(trace)))
    max_rot = max(rot_changes) if rot_changes else 0

    return trans_range.max() < trans_threshold and max_rot < np.radians(rot_threshold_deg)


def process_stream(stream, output_dir, clip_frames=81, target_fps=16,
                   height=480, width=832, stride=40, default_fov=106.26,
                   val_ratio=0.0):
    """
    Process a single player stream: extract clips with action/pose data.
    Returns list of clip metadata dicts.
    """
    stem = stream["stem"]
    print(f"  Processing: {stem}")

    # Load action data
    action_frames_list, by_frame_count = load_action_data(stream["action_path"])
    if not action_frames_list:
        print(f"    [WARN] Empty action JSON, skipping")
        return []

    # Extract video frames
    try:
        frames, video_frame_indices, source_fps = extract_video_frames(
            stream["video_path"], stream["video_fps"], target_fps
        )
    except Exception as e:
        print(f"    [ERROR] Failed to extract video: {e}")
        return []

    if len(frames) < clip_frames:
        print(f"    [WARN] Too short ({len(frames)} frames < {clip_frames}), skipping")
        return []

    # Align video frames to action data
    tf_ratio = stream["tf_ratio"]
    aligned_actions = align_video_to_action(video_frame_indices, by_frame_count, tf_ratio)

    # Determine split
    split = stream["split"]
    map_name = stream["map_name"]

    clips = []
    clip_idx = 0

    for start in range(0, len(frames) - clip_frames + 1, stride):
        end = start + clip_frames
        clip_video_frames = frames[start:end]
        clip_aligned = aligned_actions[start:end]

        # Extract poses and actions
        poses = []
        actions = []
        valid = True

        for af in clip_aligned:
            result = extract_pose_from_action_frame(af, default_fov)
            if result is None:
                valid = False
                break
            pose, action, _ = result
            poses.append(pose)
            actions.append(action)

        if not valid:
            continue

        poses = np.array(poses, dtype=np.float32)    # [N, 4, 4]
        actions = np.array(actions, dtype=np.int32)   # [N, 4]

        # Filter static clips
        if is_static_clip(poses):
            continue

        # Intrinsics (same for all frames)
        intrinsics = np.tile(
            fov_to_intrinsics(default_fov, height, width),
            (clip_frames, 1)
        )  # [N, 4]

        # Save clip
        clip_name = f"{stem}_clip{clip_idx:04d}"
        clip_dir = os.path.join(output_dir, split, "clips", clip_name)
        os.makedirs(clip_dir, exist_ok=True)

        # Save video
        video_out = os.path.join(clip_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_out, fourcc, target_fps, (width, height))
        for frame in clip_video_frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)
        out.release()

        # Save first frame
        first_frame = cv2.resize(clip_video_frames[0], (width, height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(clip_dir, "image.jpg"), first_frame)

        # Save numpy arrays
        np.save(os.path.join(clip_dir, "poses.npy"), poses)
        np.save(os.path.join(clip_dir, "action.npy"), actions)
        np.save(os.path.join(clip_dir, "intrinsics.npy"), intrinsics)

        # Generate prompt
        weapon = "rifle"
        if clip_aligned[0] is not None:
            weapon = clip_aligned[0].get("action", {}).get("weapon_slot", "rifle")

        prompt = (
            f"First-person view of a competitive CS:GO match on {map_name}. "
            f"The player is moving through the map holding a {weapon}. "
            f"Photorealistic game rendering with detailed textures, lighting effects, "
            f"and HUD elements visible."
        )
        with open(os.path.join(clip_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        clips.append({
            "clip_name": clip_name,
            "clip_path": os.path.join(split, "clips", clip_name),
            "prompt": prompt,
            "split": split,
            "map": map_name,
            "episode_id": stream["episode_id"],
            "stem": stem,
            "num_frames": clip_frames,
        })
        clip_idx += 1

    print(f"    Generated {clip_idx} clips")
    return clips


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess CSGO dataset (v3) for LingBot-World fine-tuning")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to dataset root (contains train/, test/ dirs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--clip_frames", type=int, default=81,
                        help="Frames per clip (must be 4n+1)")
    parser.add_argument("--target_fps", type=int, default=16,
                        help="Target FPS after downsampling")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    parser.add_argument("--width", type=int, default=832, help="Output video width")
    parser.add_argument("--stride", type=int, default=40,
                        help="Stride between clip starts (in downsampled frames)")
    parser.add_argument("--default_fov", type=float, default=106.26,
                        help="Default horizontal FOV in degrees")
    parser.add_argument("--skip_episodes", type=str, default="",
                        help="Comma-separated episode IDs to skip (e.g. '000004')")
    parser.add_argument("--val_split", type=str, default="",
                        help="Comma-separated episode IDs to force into val set")
    args = parser.parse_args()

    assert (args.clip_frames - 1) % 4 == 0, "clip_frames must be 4n+1"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train", "clips"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val", "clips"), exist_ok=True)

    # Parse skip/val lists
    skip_episodes = [s.strip() for s in args.skip_episodes.split(",") if s.strip()]
    val_episodes = [s.strip() for s in args.val_split.split(",") if s.strip()]

    # Discover all player streams
    streams = find_player_streams(args.input_dir, skip_episodes=skip_episodes)
    if not streams:
        print("No player streams found! Check input_dir path.")
        sys.exit(1)

    # Override split for val episodes
    if val_episodes:
        for s in streams:
            if s["episode_id"] in val_episodes:
                s["split"] = "val"

    # Process all streams
    all_clips = []
    for stream in streams:
        clips = process_stream(
            stream, args.output_dir,
            clip_frames=args.clip_frames,
            target_fps=args.target_fps,
            height=args.height,
            width=args.width,
            stride=args.stride,
            default_fov=args.default_fov,
        )
        all_clips.extend(clips)

    # Write metadata CSVs
    for split in ["train", "val"]:
        split_clips = [c for c in all_clips if c["split"] == split]
        csv_path = os.path.join(args.output_dir, f"metadata_{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "prompt", "video", "clip_path", "map", "episode_id", "stem", "num_frames"
            ])
            writer.writeheader()
            for clip in split_clips:
                writer.writerow({
                    "prompt": clip["prompt"],
                    "video": os.path.join(clip["clip_path"], "video.mp4"),
                    "clip_path": clip["clip_path"],
                    "map": clip["map"],
                    "episode_id": clip["episode_id"],
                    "stem": clip["stem"],
                    "num_frames": clip["num_frames"],
                })
        print(f"{split}: {len(split_clips)} clips -> {csv_path}")

    print(f"\nTotal: {len(all_clips)} clips processed")
    print(f"  Train: {len([c for c in all_clips if c['split'] == 'train'])}")
    print(f"  Val:   {len([c for c in all_clips if c['split'] == 'val'])}")


if __name__ == "__main__":
    main()
