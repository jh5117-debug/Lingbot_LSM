"""
CSGO Dataset Preprocessor for LingBot-World Fine-tuning
=======================================================
Converts raw CSGO dataset (videos + action JSONs + motion JSONLs) into
LingBot-compatible training clips with poses.npy, action.npy, intrinsics.npy.

Usage:
    python preprocess_csgo.py \
        --input_dir /home/nvme02/lingbot-world/datasets/datasets\ -\ poc/ \
        --output_dir /home/nvme02/lingbot-world/datasets/csgo_processed/ \
        --clip_frames 81 \
        --target_fps 16 \
        --height 480 --width 832 \
        --stride 40 \
        --val_episodes "EvilGeniuses_vs_Kari_ep5,EvilGeniuses_vs_Kari_ep6,Thunderflash_vs_WuTang_ep5,Thunderflash_vs_WuTang_ep6"
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def find_episodes(input_dir):
    """
    Walk through the CSGO dataset directory structure and find all episodes.
    Returns list of dicts with paths to video, action json, motion jsonl, episode_info.
    """
    episodes = []
    # Walk to find all episode_info.json files
    for root, dirs, files in os.walk(input_dir):
        info_files = [f for f in files if f.endswith("_episode_info.json")]
        for info_file in info_files:
            stem = info_file.replace("_episode_info.json", "")
            info_path = os.path.join(root, info_file)
            video_path = os.path.join(root, f"{stem}.mp4")
            action_path = os.path.join(root, f"{stem}.json")
            motion_path = os.path.join(root, f"{stem}_motion.jsonl")

            if not os.path.exists(video_path):
                print(f"[WARN] Video not found: {video_path}, skipping")
                continue
            if not os.path.exists(action_path):
                print(f"[WARN] Action JSON not found: {action_path}, skipping")
                continue

            with open(info_path, "r") as f:
                info = json.load(f)

            episode_id = f"{stem}"
            episodes.append({
                "episode_id": episode_id,
                "video_path": video_path,
                "action_path": action_path,
                "motion_path": motion_path if os.path.exists(motion_path) else None,
                "info": info,
                "dir": root,
            })

    print(f"Found {len(episodes)} episodes")
    return episodes


def load_action_data(action_path):
    """Load action sequence from JSON file."""
    with open(action_path, "r") as f:
        frames = json.load(f)
    return frames


def load_motion_data(motion_path, max_lines=None):
    """Load motion/camera data from JSONL file. Returns list of dicts."""
    if motion_path is None:
        return None
    data = []
    with open(motion_path, "r") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def csgo_to_pose_matrix(yaw_deg, pitch_deg, x, y, z):
    """
    Convert CSGO yaw/pitch/position to 4x4 camera-to-world matrix (OpenCV convention).

    Source engine coordinate system:
        x = forward, y = left, z = up
    OpenCV coordinate system:
        x = right, y = down, z = forward
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Build rotation: Source engine yaw is rotation around z-axis (up),
    # pitch is rotation around y-axis (left)
    # Convert to rotation matrix in Source engine coords first
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
    R_opencv = np.stack([right, -up, forward], axis=1)  # [3, 3]

    pose = np.eye(4)
    pose[:3, :3] = R_opencv
    pose[:3, 3] = np.array([x, y, z])
    return pose


def fov_to_intrinsics(fov_deg, height, width):
    """
    Convert horizontal FOV to camera intrinsics [fx, fy, cx, cy].
    """
    fov_rad = np.radians(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def extract_video_frames(video_path, target_fps, source_tick_rate=128):
    """
    Extract frames from video, downsampling from source_tick_rate to target_fps.
    Returns list of frames (numpy arrays in BGR format from cv2).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # If video FPS is available and differs from tick rate, use it
    if video_fps > 0 and abs(video_fps - source_tick_rate) > 1:
        # Video may already be at a different FPS
        actual_source_fps = video_fps
    else:
        actual_source_fps = source_tick_rate

    # Calculate frame skip interval
    skip = max(1, round(actual_source_fps / target_fps))

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
    return frames, frame_indices, actual_source_fps


def process_episode(episode, output_dir, clip_frames=81, target_fps=16,
                    height=480, width=832, stride=40, is_val=False):
    """
    Process a single episode: extract clips with action/pose data.
    Returns list of clip metadata dicts.
    """
    print(f"  Processing: {episode['episode_id']}")

    # Load action data
    action_frames = load_action_data(episode["action_path"])

    # Load motion data for camera FOV (if available)
    motion_data = None
    default_fov = 106.26  # typical CSGO FOV from dataset
    if episode["motion_path"]:
        try:
            motion_data = load_motion_data(episode["motion_path"], max_lines=100)
            if motion_data and "camera" in motion_data[0]:
                default_fov = motion_data[0]["camera"].get("fov", default_fov)
        except Exception as e:
            print(f"    [WARN] Could not load motion data: {e}")

    # Extract video frames
    try:
        frames, frame_indices, source_fps = extract_video_frames(
            episode["video_path"], target_fps
        )
    except Exception as e:
        print(f"    [ERROR] Failed to extract video: {e}")
        return []

    if len(frames) < clip_frames:
        print(f"    [WARN] Episode too short ({len(frames)} frames < {clip_frames}), skipping")
        return []

    # Map video frame indices to action frame indices
    # The action JSON has entries at source tick rate; frame_indices maps to those
    skip = max(1, round(source_fps / target_fps))

    clips = []
    split = "val" if is_val else "train"
    clip_idx = 0

    for start in range(0, len(frames) - clip_frames + 1, stride):
        end = start + clip_frames
        clip_video_frames = frames[start:end]
        clip_action_indices = frame_indices[start:end]

        # Extract poses and actions for this clip
        poses = []
        actions = []
        valid = True

        for vid_idx in clip_action_indices:
            # Find corresponding action frame
            if vid_idx // 2 >= len(action_frames):
                valid = False
                break
            af = action_frames[min(vid_idx // 2, len(action_frames) - 1)]

            # Build pose matrix
            # Fix: convert 0-360 degree format to ±180
            yaw = af.get("yaw", 0.0)
            pitch = af.get("pitch", 0.0)
            if yaw > 180:
                yaw -= 360
            if pitch > 180:
                pitch -= 360

            # Fix: use camera_position (eye height) instead of foot position (x, y, z)
            cam_pos = af.get("camera_position", None)
            if cam_pos is not None:
                x, y, z = cam_pos[0], cam_pos[1], cam_pos[2]
            else:
                x = af.get("x", 0.0)
                y = af.get("y", 0.0)
                z = af.get("z", 0.0)

            pose = csgo_to_pose_matrix(yaw, pitch, x, y, z)
            poses.append(pose)

            # Build action vector [forward, back, left, right] as 4D integers
            act = af.get("action", {})
            fwd = int(bool(act.get("forward", False)))
            back = int(bool(act.get("back", False)))
            left = int(bool(act.get("left", False)))
            right = int(bool(act.get("right", False)))
            actions.append([fwd, back, left, right])

        if not valid:
            continue

        poses = np.array(poses, dtype=np.float32)   # [N, 4, 4]
        actions = np.array(actions, dtype=np.int32)  # [N, 4]: [forward, back, left, right]

        # Fix: filter out static clips where camera barely moves
        translations = poses[:, :3, 3]  # [N, 3]
        trans_range = translations.max(axis=0) - translations.min(axis=0)
        rotations = poses[:, :3, :3]  # [N, 3, 3]
        # Check rotation change via trace of relative rotation
        rot_changes = []
        for fi in range(1, len(rotations)):
            rel_rot = rotations[fi] @ rotations[fi-1].T
            rot_changes.append(abs(np.arccos(np.clip((np.trace(rel_rot) - 1) / 2, -1, 1))))
        max_rot_change = max(rot_changes) if rot_changes else 0
        if trans_range.max() < 1e-3 and max_rot_change < np.radians(1.0):
            # Skip clips with almost no camera movement
            continue

        # Intrinsics (same for all frames in a clip)
        intrinsics = np.tile(
            fov_to_intrinsics(default_fov, height, width),
            (clip_frames, 1)
        )  # [N, 4]

        # Save clip
        clip_name = f"{episode['episode_id']}_clip{clip_idx:04d}"
        clip_dir = os.path.join(output_dir, split, "clips", clip_name)
        os.makedirs(clip_dir, exist_ok=True)

        # Save video as mp4
        video_out_path = os.path.join(clip_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_out_path, fourcc, target_fps, (width, height))
        for frame in clip_video_frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)
        out.release()

        # Save first frame as image (for I2V inference)
        first_frame = cv2.resize(clip_video_frames[0], (width, height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(clip_dir, "image.jpg"), first_frame)

        # Save numpy arrays
        np.save(os.path.join(clip_dir, "poses.npy"), poses)
        np.save(os.path.join(clip_dir, "action.npy"), actions)
        np.save(os.path.join(clip_dir, "intrinsics.npy"), intrinsics)

        # Generate prompt
        info = episode["info"]
        map_name = "unknown"
        # Try to detect map from directory path
        path_lower = episode["dir"].lower()
        if "nuke" in path_lower:
            map_name = "de_nuke"
        elif "mirage" in path_lower:
            map_name = "de_mirage"

        weapon = "rifle"
        # Try to get weapon from first action frame
        if clip_action_indices[0] < len(action_frames):
            weapon = action_frames[min(clip_action_indices[0] // 2, len(action_frames) - 1)].get("action", {}).get("weapon_slot", "rifle")

        prompt = (
            f"First-person view of a competitive CS:GO match on {map_name}. "
            f"The player is moving through the map holding a {weapon}. "
            f"Photorealistic game rendering with detailed textures, lighting effects, "
            f"and HUD elements visible."
        )

        # Save prompt
        with open(os.path.join(clip_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        clips.append({
            "clip_name": clip_name,
            "clip_path": os.path.join(split, "clips", clip_name),
            "prompt": prompt,
            "split": split,
            "map": map_name,
            "episode_id": episode["episode_id"],
            "num_frames": clip_frames,
        })
        clip_idx += 1

    print(f"    Generated {clip_idx} clips")
    return clips


def main():
    parser = argparse.ArgumentParser(description="Preprocess CSGO dataset for LingBot-World fine-tuning")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw CSGO dataset root")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--clip_frames", type=int, default=81, help="Frames per clip (4n+1)")
    parser.add_argument("--target_fps", type=int, default=16, help="Target FPS after downsampling")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    parser.add_argument("--width", type=int, default=832, help="Output video width")
    parser.add_argument("--stride", type=int, default=40, help="Stride between clip starts (frames)")
    parser.add_argument("--val_episodes", type=str, default="",
                        help="Comma-separated list of episode ID substrings to use as validation")
    args = parser.parse_args()

    assert (args.clip_frames - 1) % 4 == 0, "clip_frames must be 4n+1"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train", "clips"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val", "clips"), exist_ok=True)

    # Find all episodes
    episodes = find_episodes(args.input_dir)
    if not episodes:
        print("No episodes found! Check input_dir path.")
        sys.exit(1)

    # Parse val episode list
    val_patterns = [p.strip() for p in args.val_episodes.split(",") if p.strip()]

    all_clips = []
    for ep in episodes:
        is_val = any(pat in ep["episode_id"] for pat in val_patterns) if val_patterns else False
        clips = process_episode(
            ep, args.output_dir,
            clip_frames=args.clip_frames,
            target_fps=args.target_fps,
            height=args.height,
            width=args.width,
            stride=args.stride,
            is_val=is_val,
        )
        all_clips.extend(clips)

    # Write metadata CSVs
    for split in ["train", "val"]:
        split_clips = [c for c in all_clips if c["split"] == split]
        csv_path = os.path.join(args.output_dir, f"metadata_{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "video", "clip_path", "map", "episode_id", "num_frames"])
            writer.writeheader()
            for clip in split_clips:
                writer.writerow({
                    "prompt": clip["prompt"],
                    "video": os.path.join(clip["clip_path"], "video.mp4"),
                    "clip_path": clip["clip_path"],
                    "map": clip["map"],
                    "episode_id": clip["episode_id"],
                    "num_frames": clip["num_frames"],
                })
        print(f"{split}: {len(split_clips)} clips → {csv_path}")

    print(f"\nTotal: {len(all_clips)} clips processed")
    print(f"  Train: {len([c for c in all_clips if c['split'] == 'train'])}")
    print(f"  Val:   {len([c for c in all_clips if c['split'] == 'val'])}")


if __name__ == "__main__":
    main()
