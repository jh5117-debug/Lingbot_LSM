"""
Visibility computation for cross-player attention.

Determines which players can see each other at each frame based on:
1. FOV check: Is player j within player i's field of view?
2. Occlusion check: Is the line of sight between i and j blocked by geometry?

When 3D mesh is not available, falls back to FOV-only check (no occlusion).
"""

import math
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


class VisibilityComputer:
    """
    Computes pairwise visibility between players.

    Output: visibility_matrix[i][j] = 1 if player j is visible from player i's POV.
    Note: visibility is NOT symmetric (i sees j doesn't imply j sees i).
    """

    def __init__(
        self,
        fov_degrees: float = 90.0,
        max_distance: float = 5000.0,
        mesh_path: Optional[str] = None,
    ):
        """
        Args:
            fov_degrees: Horizontal field of view in degrees (CSGO default ~90).
            max_distance: Maximum visibility distance in game units.
            mesh_path: Optional path to map .obj for occlusion testing.
        """
        self.fov_half = math.radians(fov_degrees / 2.0)
        self.max_distance = max_distance
        self.mesh = None

        if mesh_path is not None:
            if not TRIMESH_AVAILABLE:
                logging.warning("trimesh not available, occlusion checks disabled")
            else:
                logging.info(f"Loading mesh for visibility: {mesh_path}")
                self.mesh = trimesh.load(mesh_path, force='mesh')

    def compute_frame(
        self,
        players: List[Dict],
    ) -> np.ndarray:
        """
        Compute visibility matrix for a single frame.

        Args:
            players: List of player dicts, each containing:
                {
                    "x": float, "y": float, "z": float,
                    "yaw": float,  # degrees, 0=east, counter-clockwise
                    "alive": bool,
                }

        Returns:
            vis_matrix: [N, N] binary numpy array.
                vis_matrix[i, j] = 1 means player i can see player j.
        """
        n = len(players)
        vis = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            if not players[i].get("alive", True):
                continue
            for j in range(n):
                if i == j:
                    continue
                if not players[j].get("alive", True):
                    continue

                vis[i, j] = self._check_visibility(players[i], players[j])

        return vis

    def compute_sequence(
        self,
        players_sequence: List[List[Dict]],
    ) -> np.ndarray:
        """
        Compute visibility matrices for a frame sequence.

        Args:
            players_sequence: [T] list, each element is a list of N player dicts.

        Returns:
            vis_matrices: [T, N, N] binary numpy array.
        """
        T = len(players_sequence)
        N = len(players_sequence[0])
        vis = np.zeros((T, N, N), dtype=np.float32)

        for t in range(T):
            vis[t] = self.compute_frame(players_sequence[t])

        return vis

    def _check_visibility(self, viewer: Dict, target: Dict) -> float:
        """Check if viewer can see target. Returns 1.0 if visible, 0.0 otherwise."""
        # Vector from viewer to target
        dx = target["x"] - viewer["x"]
        dy = target["y"] - viewer["y"]
        dz = target["z"] - viewer["z"]

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > self.max_distance or dist < 1e-6:
            return 0.0

        # FOV check (horizontal plane)
        angle_to_target = math.atan2(dy, dx)  # radians
        viewer_yaw = math.radians(viewer["yaw"])

        # Angular difference (wrapped to [-pi, pi])
        angle_diff = angle_to_target - viewer_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > self.fov_half:
            return 0.0

        # Occlusion check via ray casting (if mesh available)
        if self.mesh is not None:
            if self._is_occluded(viewer, target):
                return 0.0

        return 1.0

    def _is_occluded(self, viewer: Dict, target: Dict) -> bool:
        """Check if line of sight is blocked by geometry."""
        origin = np.array([[viewer["x"], viewer["y"], viewer["z"]]])
        direction = np.array([[
            target["x"] - viewer["x"],
            target["y"] - viewer["y"],
            target["z"] - viewer["z"],
        ]])

        dist = np.linalg.norm(direction)
        direction = direction / dist

        locations, ray_idx, tri_idx = self.mesh.ray.intersects_location(
            ray_origins=origin,
            ray_directions=direction,
        )

        if len(locations) == 0:
            return False

        # Check if any hit is closer than the target
        hit_distances = np.linalg.norm(locations - origin, axis=1)
        # Small tolerance to avoid self-intersection
        min_hit = hit_distances[hit_distances > 10.0].min() if (hit_distances > 10.0).any() else float('inf')

        return min_hit < (dist - 10.0)

    def get_visible_players(
        self,
        vis_matrix: np.ndarray,
        player_idx: int,
    ) -> List[int]:
        """Get indices of players visible to player_idx."""
        return list(np.where(vis_matrix[player_idx] > 0.5)[0])


def precompute_visibility(
    episode_dir: str,
    num_players: int,
    action_files: List[str],
    output_path: str,
    mesh_path: Optional[str] = None,
    fov_degrees: float = 90.0,
    subsample: int = 4,
):
    """
    Pre-compute visibility matrices for an entire episode.
    Loads action data for all players and computes per-frame visibility.

    Args:
        episode_dir: Directory containing player action files.
        num_players: Number of players.
        action_files: List of paths to action JSON files (one per player).
        output_path: Where to save the visibility .npy file.
        mesh_path: Optional map mesh for occlusion checks.
        fov_degrees: Horizontal FOV.
        subsample: Only compute every N-th frame (to save time).

    Saves:
        vis_matrices: [T // subsample, N, N] numpy array.
    """
    import json

    computer = VisibilityComputer(
        fov_degrees=fov_degrees,
        mesh_path=mesh_path,
    )

    # Load all players' action data
    all_actions = []
    for af_path in action_files:
        with open(af_path, 'r') as f:
            actions = json.load(f)
        all_actions.append(actions)

    # Determine frame count (minimum across all players)
    min_frames = min(len(a) for a in all_actions)
    num_output_frames = (min_frames + subsample - 1) // subsample

    vis_matrices = np.zeros((num_output_frames, num_players, num_players), dtype=np.float32)

    for t_out, t_in in enumerate(range(0, min_frames, subsample)):
        players = []
        for p_idx in range(num_players):
            frame = all_actions[p_idx][t_in]

            # Extract position (prefer render_transform, fallback to camera_position)
            rt = frame.get("render_transform")
            cam_pos = frame.get("camera_position")
            if rt and rt.get("x") is not None:
                x, y = rt["x"], rt["y"]
                z = cam_pos[2] if cam_pos else rt.get("z", 0)
            elif cam_pos:
                x, y, z = cam_pos[0], cam_pos[1], cam_pos[2]
            else:
                x = frame.get("x", 0)
                y = frame.get("y", 0)
                z = frame.get("z", 0)

            # Extract yaw (prefer camera_rotation)
            cam_rot = frame.get("camera_rotation")
            if cam_rot:
                yaw = cam_rot[2]
            else:
                yaw = frame.get("yaw", 0)

            health = frame.get("health", 100)

            players.append({
                "x": x, "y": y, "z": z,
                "yaw": yaw,
                "alive": health > 0,
            })

        vis_matrices[t_out] = computer.compute_frame(players)

    np.save(output_path, vis_matrices)
    logging.info(f"Saved visibility matrices [{num_output_frames}, {num_players}, {num_players}] to {output_path}")
    return vis_matrices
