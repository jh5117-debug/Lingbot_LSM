"""
Stage 2 Dataset: Multi-player episode-grouped data loading.

For Stage 2, we need to load multiple players from the same episode
simultaneously, along with BEV maps and visibility matrices.

Phase 2a (BEV-only): loads single player + BEV, like Stage 1 but with BEV.
Phase 2b (cross-player): loads player groups (2-3 visible players).
"""

import csv
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .bev_builder import (
    DynamicBEVBuilder, estimate_bounds_from_positions,
    NUM_DYNAMIC_CHANNELS, NUM_STATIC_CHANNELS,
)


class Stage2Dataset(Dataset):
    """
    Multi-player dataset for Stage 2 training.

    Groups clips by episode so we can load multiple players from the same
    time window. Each sample returns one "anchor" player's video/actions
    plus BEV map and optionally visible players' data.

    Directory structure (preprocessed):
        dataset_dir/
          metadata_train.csv    (has: clip_path, episode_id, stem, ...)
          train/clips/
            clip_NNNN/          (per-player clip)
              video.mp4
              poses.npy         [F, 4, 4]
              action.npy        [F, 7]  (rays_d[3] + wasd[4])
              intrinsics.npy    [F, 3, 3]

    Additional Stage 2 data (from preprocess_stage2.py):
        stage2_dir/
          static_bev.npy                [4, 256, 256]
          static_bev_bounds.json
          episodes/
            Ep_NNNNNN/
              dynamic_bev.npy           [F, 6, 256, 256] (per episode, per player)
              visibility.npy            [F, N, N]
              player_states.json        per-frame all-player states
    """

    def __init__(
        self,
        dataset_dir: str,
        stage2_dir: str,
        split: str = "train",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        bev_size: int = 256,
        phase: str = "2a",
        num_context_players: int = 2,
        repeat: int = 1,
    ):
        """
        Args:
            dataset_dir: Root of preprocessed Stage 1 data.
            stage2_dir: Root of Stage 2 preprocessed data (BEV, visibility).
            split: "train" or "val".
            num_frames: Frames per clip.
            height, width: Video resolution.
            bev_size: BEV spatial resolution.
            phase: "2a" (BEV-only) or "2b" (BEV + cross-player).
            num_context_players: Max visible players to include (Phase 2b).
            repeat: Dataset repeat factor.
        """
        self.dataset_dir = dataset_dir
        self.stage2_dir = stage2_dir
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.bev_size = bev_size
        self.phase = phase
        self.num_context_players = num_context_players
        self.repeat = repeat

        # Load metadata
        csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {csv_path}")

        # Group clips by episode for multi-player loading
        self.episode_clips = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.episode_clips[s["episode_id"]].append(i)

        # Parse player info from stem (e.g., "Ep_000005_team_2_player_0000_inst_000")
        for s in self.samples:
            stem = s["stem"]
            m = re.search(r'team_(\d+)_player_(\d+)', stem)
            if m:
                s["team_id"] = int(m.group(1))
                s["player_idx"] = int(m.group(2))
            else:
                s["team_id"] = 0
                s["player_idx"] = 0

        # Load static BEV if available
        static_bev_path = os.path.join(stage2_dir, "static_bev.npy")
        if os.path.exists(static_bev_path):
            self.static_bev = np.load(static_bev_path)
            bounds_path = os.path.join(stage2_dir, "static_bev_bounds.json")
            with open(bounds_path) as f:
                self.bev_bounds = json.load(f)
            logging.info(f"Loaded static BEV: {self.static_bev.shape}")
        else:
            self.static_bev = None
            self.bev_bounds = None
            logging.warning("Static BEV not found, will use zeros")

        # Dynamic BEV builder
        self.dynamic_bev_builder = DynamicBEVBuilder(bev_size=bev_size)
        if self.bev_bounds:
            self.dynamic_bev_builder.set_map_bounds(self.bev_bounds)

        logging.info(
            f"Stage2Dataset: {len(self.samples)} {split} samples, "
            f"{len(self.episode_clips)} episodes, "
            f"phase={phase}, repeat={repeat}"
        )

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.samples)
        sample = self.samples[idx]
        episode_id = sample["episode_id"]

        # Load anchor player data (same as Stage 1)
        anchor_data = self._load_player_clip(sample)

        # Build BEV map for this clip
        bev_map = self._build_bev(sample, episode_id)

        result = {
            "video": anchor_data["video"],
            "prompt": anchor_data["prompt"],
            "poses": anchor_data["poses"],
            "actions": anchor_data["actions"],
            "intrinsics": anchor_data["intrinsics"],
            "bev_map": torch.from_numpy(bev_map).float(),
            "episode_id": episode_id,
            "player_idx": sample["player_idx"],
            "team_id": sample["team_id"],
        }

        # Phase 2b: also load context players
        if self.phase == "2b":
            context = self._load_context_players(sample, episode_id)
            result["context_players"] = context

        return result

    def _load_player_clip(self, sample: dict) -> dict:
        """Load video, poses, actions, intrinsics for a single player clip."""
        clip_dir = os.path.join(self.dataset_dir, sample["clip_path"])

        # Load video
        video_path = os.path.join(clip_dir, "video.mp4")
        frames = []
        cap = cv2.VideoCapture(video_path)
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_LANCZOS4)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(frame)
        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        video = torch.stack(frames, dim=1)  # [3, F, H, W]

        # Load numpy arrays
        poses = np.load(os.path.join(clip_dir, "poses.npy"))
        actions = np.load(os.path.join(clip_dir, "action.npy"))
        intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))

        poses = self._pad_or_truncate(poses, self.num_frames)
        actions = self._pad_or_truncate(actions, self.num_frames)
        intrinsics = self._pad_or_truncate(intrinsics, self.num_frames)

        return {
            "video": video,
            "prompt": sample["prompt"],
            "poses": torch.from_numpy(poses).float(),
            "actions": torch.from_numpy(actions).float(),
            "intrinsics": torch.from_numpy(intrinsics).float(),
        }

    def _build_bev(self, sample: dict, episode_id: str) -> np.ndarray:
        """
        Build combined BEV (static + dynamic) for this clip.

        Returns:
            bev: [C_total, bev_size, bev_size]
        """
        # Try loading pre-computed dynamic BEV
        dyn_bev_path = os.path.join(
            self.stage2_dir, "episodes", f"Ep_{episode_id}",
            f"dynamic_bev_{sample['stem']}.npy"
        )

        if os.path.exists(dyn_bev_path):
            # Load pre-computed dynamic BEV (already frame-averaged or specific clip)
            dynamic_bev = np.load(dyn_bev_path)
            # Take mean across frames for a single BEV conditioning
            if dynamic_bev.ndim == 4:  # [F, C, H, W]
                dynamic_bev = dynamic_bev.mean(axis=0)  # [C, H, W]
        else:
            # Fallback: zero dynamic BEV (will learn from static BEV only)
            dynamic_bev = np.zeros(
                (NUM_DYNAMIC_CHANNELS, self.bev_size, self.bev_size),
                dtype=np.float32
            )

        # Combine static + dynamic
        if self.static_bev is not None:
            bev = np.concatenate([self.static_bev, dynamic_bev], axis=0)
        else:
            # No static BEV: use zeros for static channels
            static_zeros = np.zeros(
                (4, self.bev_size, self.bev_size), dtype=np.float32
            )
            bev = np.concatenate([static_zeros, dynamic_bev], axis=0)

        return bev  # [10, bev_size, bev_size]

    def _load_context_players(
        self,
        anchor_sample: dict,
        episode_id: str,
    ) -> dict:
        """
        Load visible context players for Phase 2b cross-player attention.

        Returns dict with:
            "videos": list of [3, F, H, W] tensors (context player videos)
            "poses": list of [F, 4, 4] tensors
            "actions": list of [F, 7] tensors
            "intrinsics": list of [F, 3, 3] tensors
            "visibility_weights": [num_context] visibility scores
        """
        # Load pre-computed visibility matrix
        vis_path = os.path.join(
            self.stage2_dir, "episodes", f"Ep_{episode_id}", "visibility.npy"
        )

        # Find clips from same episode (excluding anchor)
        episode_clip_indices = self.episode_clips[episode_id]
        anchor_player_idx = anchor_sample["player_idx"]

        candidate_clips = []
        for ci in episode_clip_indices:
            s = self.samples[ci]
            if s["player_idx"] != anchor_player_idx:
                candidate_clips.append(s)

        if not candidate_clips:
            return self._empty_context()

        # If we have pre-computed visibility, use it to rank candidates
        if os.path.exists(vis_path):
            vis = np.load(vis_path)  # [T, N, N]
            # Average visibility over time
            avg_vis = vis.mean(axis=0)  # [N, N]
            # Score each candidate by visibility from anchor
            scored = []
            for s in candidate_clips:
                pidx = s["player_idx"]
                if pidx < avg_vis.shape[0] and anchor_player_idx < avg_vis.shape[0]:
                    score = avg_vis[anchor_player_idx, pidx]
                else:
                    score = 0.0
                scored.append((score, s))
            # Sort by visibility score descending
            scored.sort(key=lambda x: -x[0])
            # Filter to actually visible players
            scored = [(sc, s) for sc, s in scored if sc > 0.1]
        else:
            # No visibility data: randomly sample from same-episode players
            scored = [(1.0, s) for s in candidate_clips]

        # Take top N context players
        selected = scored[:self.num_context_players]

        if not selected:
            return self._empty_context()

        context = {
            "videos": [],
            "poses": [],
            "actions": [],
            "intrinsics": [],
            "visibility_weights": [],
        }

        for score, s in selected:
            data = self._load_player_clip(s)
            context["videos"].append(data["video"])
            context["poses"].append(data["poses"])
            context["actions"].append(data["actions"])
            context["intrinsics"].append(data["intrinsics"])
            context["visibility_weights"].append(score)

        context["visibility_weights"] = torch.tensor(
            context["visibility_weights"], dtype=torch.float32
        )

        return context

    def _empty_context(self) -> dict:
        """Return empty context when no visible players exist."""
        return {
            "videos": [],
            "poses": [],
            "actions": [],
            "intrinsics": [],
            "visibility_weights": torch.zeros(0),
        }

    def _pad_or_truncate(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) >= target_len:
            return arr[:target_len]
        rep_shape = (target_len - len(arr),) + (1,) * (arr.ndim - 1)
        pad = np.tile(arr[-1:], rep_shape)
        return np.concatenate([arr, pad], axis=0)


# ============================================================
# Collate function for Stage 2
# ============================================================

def stage2_collate_fn(batch):
    """
    Custom collate for Stage 2 that handles variable-length context players.
    Since we use batch_size=1 (like Stage 1), this just unwraps the single sample.
    """
    if isinstance(batch, list) and len(batch) == 1:
        return batch[0]
    return batch
