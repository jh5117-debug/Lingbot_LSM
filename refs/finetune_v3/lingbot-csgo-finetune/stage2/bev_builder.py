"""
BEV (Bird's Eye View) map construction for CSGO.

Design principle: ALL channels must be available at inference time.
  - Static BEV: computed once per map (from navmesh or 3D mesh)
  - Dynamic BEV: computed per-frame from player poses only

Static BEV Channels (4):
  0: Walkability (from navmesh areas or mesh normals)
  1: Ground height (normalized)
  2: Cover/hiding spots (from navmesh hiding_spots or mesh ray-cast)
  3: Semantic region encoding (from navmesh place_name, 26 regions -> normalized ID)

Dynamic BEV Channels (3):
  0: Friendly player positions (Gaussian heatmap)
  1: Enemy player positions (Gaussian heatmap)
  2: Current player position + orientation (arrow marker)

Total: 4 static + 3 dynamic = 7 channels

REMOVED (not available at inference):
  - Active projectiles (smoke/flash/fire) — requires world_events
  - Shooting lines — requires fire action (not in action.npy)
  - Health change markers — requires health field
"""

import os
import json
import logging
import math
from typing import Optional, Tuple, List, Dict

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


NUM_STATIC_CHANNELS = 4
NUM_DYNAMIC_CHANNELS = 3
NUM_BEV_CHANNELS = NUM_STATIC_CHANNELS + NUM_DYNAMIC_CHANNELS  # 7


# ============================================================
# Static BEV Builder
# ============================================================

class StaticBEVBuilder:
    """
    Constructs a static BEV feature map from navmesh or 3D mesh.
    Computed once per map, cached to disk for reuse.

    Channels:
      0: Walkability (1=walkable area, 0=not)
      1: Normalized ground height
      2: Cover / hiding spots
      3: Semantic region encoding (place_name -> normalized ID)
    """

    PLAYER_HEIGHT = 64.0  # CSGO standing eye height in game units

    def __init__(self, bev_size: int = 256):
        self.bev_size = bev_size
        self.map_bounds = None
        self.cell_size_x = None
        self.cell_size_y = None

    # ----------------------------------------------------------------
    # Method 1: Build from navmesh (PREFERRED — fast, no trimesh needed)
    # ----------------------------------------------------------------

    def build_from_navmesh(
        self,
        navmesh_path: str,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build static BEV from navmesh.json — the engine's own navigation data.

        Advantages over mesh ray-cast:
          - Walkability is engine-authoritative (not approximated from normals)
          - Includes semantic region names (26 place_names on de_dust2)
          - Includes hiding spots (371 positions on de_dust2)
          - Orders of magnitude faster (no ray-casting)
          - No trimesh dependency

        Args:
            navmesh_path: Path to navmesh.json
            save_path: If provided, save the BEV as .npy

        Returns:
            bev: [4, bev_size, bev_size] numpy array
        """
        logging.info(f"Loading navmesh from {navmesh_path}...")
        with open(navmesh_path, 'r') as f:
            navmesh = json.load(f)

        areas = navmesh["areas"]  # dict: area_id -> area_data
        place_names = navmesh.get("place_names", [])

        # Build place_name -> normalized ID mapping
        # Sort alphabetically for consistency across runs
        unique_places = sorted(set(place_names))
        if "" not in unique_places:
            unique_places = [""] + unique_places  # ID 0 = unknown/unlabeled
        place_to_id = {name: i / max(len(unique_places) - 1, 1)
                       for i, name in enumerate(unique_places)}
        logging.info(f"Navmesh: {len(areas)} areas, {len(unique_places)} place types, "
                     f"map={navmesh.get('map_name', '?')}")

        # Determine map bounds from all area corners
        all_x, all_y, all_z = [], [], []
        for area in areas.values():
            nw = area["nw_corner"]
            se = area["se_corner"]
            all_x.extend([nw[0], se[0]])
            all_y.extend([nw[1], se[1]])
            all_z.extend([nw[2], se[2], area.get("ne_z", nw[2]), area.get("sw_z", se[2])])

        padding = 100.0
        x_min, x_max = min(all_x) - padding, max(all_x) + padding
        y_min, y_max = min(all_y) - padding, max(all_y) + padding
        z_min, z_max = min(all_z), max(all_z)

        self.map_bounds = {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max,
        }
        self.cell_size_x = (x_max - x_min) / self.bev_size
        self.cell_size_y = (y_max - y_min) / self.bev_size

        logging.info(f"Map bounds: X[{x_min:.0f}, {x_max:.0f}] "
                     f"Y[{y_min:.0f}, {y_max:.0f}] Z[{z_min:.0f}, {z_max:.0f}]")

        bev = np.zeros((NUM_STATIC_CHANNELS, self.bev_size, self.bev_size), dtype=np.float32)
        z_range = max(z_max - z_min, 1.0)

        for area in areas.values():
            nw = area["nw_corner"]
            se = area["se_corner"]

            # Convert area rectangle to grid indices
            ix_min = int((min(nw[0], se[0]) - x_min) / self.cell_size_x)
            ix_max = int((max(nw[0], se[0]) - x_min) / self.cell_size_x)
            iy_min = int((min(nw[1], se[1]) - y_min) / self.cell_size_y)
            iy_max = int((max(nw[1], se[1]) - y_min) / self.cell_size_y)

            # Clamp to grid
            ix_min = max(0, min(ix_min, self.bev_size - 1))
            ix_max = max(0, min(ix_max, self.bev_size - 1))
            iy_min = max(0, min(iy_min, self.bev_size - 1))
            iy_max = max(0, min(iy_max, self.bev_size - 1))

            # Channel 0: Walkability — navmesh area exists = walkable
            bev[0, iy_min:iy_max+1, ix_min:ix_max+1] = 1.0

            # Channel 1: Ground height — bilinear interpolation of 4 corner z values
            z_nw, z_se = nw[2], se[2]
            z_ne = area.get("ne_z", z_nw)
            z_sw = area.get("sw_z", z_se)
            avg_z = (z_nw + z_se + z_ne + z_sw) / 4.0
            height_val = (avg_z - z_min) / z_range
            # Use max to handle overlapping areas (keep highest)
            bev[1, iy_min:iy_max+1, ix_min:ix_max+1] = np.maximum(
                bev[1, iy_min:iy_max+1, ix_min:ix_max+1], height_val
            )

            # Channel 3: Semantic region encoding
            place = area.get("place_name", "")
            region_id = place_to_id.get(place, 0.0)
            bev[3, iy_min:iy_max+1, ix_min:ix_max+1] = region_id

            # Channel 2: Hiding spots
            for spot in area.get("hiding_spots", []):
                sx = int((spot["x"] - x_min) / self.cell_size_x)
                sy = int((spot["y"] - y_min) / self.cell_size_y)
                if 0 <= sx < self.bev_size and 0 <= sy < self.bev_size:
                    # Draw small Gaussian around hiding spot
                    self._stamp_hiding_spot(bev[2], sx, sy)

        logging.info(f"Navmesh BEV built: walkable cells="
                     f"{int(bev[0].sum())}/{self.bev_size**2}, "
                     f"hiding spots painted={int((bev[2] > 0).sum())} cells")

        if save_path:
            np.save(save_path, bev)
            bounds_path = save_path.replace('.npy', '_bounds.json')
            with open(bounds_path, 'w') as f:
                json.dump(self.map_bounds, f)
            # Also save place_to_id mapping for reference
            mapping_path = save_path.replace('.npy', '_place_ids.json')
            with open(mapping_path, 'w') as f:
                json.dump(place_to_id, f, indent=2)
            logging.info(f"Saved static BEV to {save_path}")

        return bev

    def _stamp_hiding_spot(self, channel: np.ndarray, cx: int, cy: int, radius: int = 3):
        """Draw a soft Gaussian dot at a hiding spot location."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= radius:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < self.bev_size and 0 <= py < self.bev_size:
                        intensity = 1.0 - dist / (radius + 1)
                        channel[py, px] = max(channel[py, px], intensity)

    # ----------------------------------------------------------------
    # Method 2: Build from 3D mesh (fallback when navmesh unavailable)
    # ----------------------------------------------------------------

    def build_from_mesh(
        self,
        map_mesh_path: str,
        static_props_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build static BEV from 3D mesh files. Slower but works without navmesh.

        Channels:
          0: Walkability (from surface normals)
          1: Normalized ground height
          2: Cover at player height (horizontal ray-cast)
          3: Zeros (no semantic info from mesh)

        Returns:
            bev: [4, bev_size, bev_size] numpy array
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for mesh-based BEV construction. "
                              "Install with: pip install trimesh")

        logging.info(f"Loading map mesh from {map_mesh_path}...")
        mesh = trimesh.load(map_mesh_path, force='mesh')
        vertices = np.array(mesh.vertices)

        x_min, y_min, z_min = vertices.min(axis=0)
        x_max, y_max, z_max = vertices.max(axis=0)

        padding = 100.0
        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding

        self.map_bounds = {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max,
        }
        self.cell_size_x = (x_max - x_min) / self.bev_size
        self.cell_size_y = (y_max - y_min) / self.bev_size

        bev = np.zeros((NUM_STATIC_CHANNELS, self.bev_size, self.bev_size), dtype=np.float32)

        logging.info("Computing height map and walkability via ray-cast...")
        bev[1], bev[0] = self._compute_height_and_walkability(mesh)

        logging.info("Computing cover map...")
        bev[2] = self._compute_cover_map(mesh)

        # Channel 3 (semantic region): zeros when built from mesh
        # (no place_name info available)

        if save_path:
            np.save(save_path, bev)
            bounds_path = save_path.replace('.npy', '_bounds.json')
            with open(bounds_path, 'w') as f:
                json.dump(self.map_bounds, f)
            logging.info(f"Saved static BEV to {save_path}")

        return bev

    def _compute_height_and_walkability(self, mesh) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ground height and walkability via downward ray casting."""
        height_map = np.zeros((self.bev_size, self.bev_size), dtype=np.float32)
        walk_map = np.zeros((self.bev_size, self.bev_size), dtype=np.float32)

        bounds = self.map_bounds
        z_top = bounds["z_max"] + 100.0

        xs = np.linspace(bounds["x_min"], bounds["x_max"], self.bev_size, endpoint=False)
        ys = np.linspace(bounds["y_min"], bounds["y_max"], self.bev_size, endpoint=False)
        xs += self.cell_size_x / 2
        ys += self.cell_size_y / 2

        ray_origins = []
        ray_directions = []
        for iy in range(self.bev_size):
            for ix in range(self.bev_size):
                ray_origins.append([xs[ix], ys[iy], z_top])
                ray_directions.append([0, 0, -1])

        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )

        if len(locations) > 0:
            for loc, ray_idx, tri_idx in zip(locations, index_ray, index_tri):
                iy = ray_idx // self.bev_size
                ix = ray_idx % self.bev_size

                z_hit = loc[2]
                if z_hit > height_map[iy, ix]:
                    height_map[iy, ix] = z_hit
                    face_normal = mesh.face_normals[tri_idx]
                    if face_normal[2] > 0.7:
                        walk_map[iy, ix] = 1.0

        z_range = bounds["z_max"] - bounds["z_min"]
        if z_range > 0:
            height_map = (height_map - bounds["z_min"]) / z_range

        return height_map, walk_map

    def _compute_cover_map(self, mesh) -> np.ndarray:
        """Compute cover at player standing height via horizontal ray-cast."""
        cover_map = np.zeros((self.bev_size, self.bev_size), dtype=np.float32)
        bounds = self.map_bounds

        xs = np.linspace(bounds["x_min"], bounds["x_max"], self.bev_size, endpoint=False)
        ys = np.linspace(bounds["y_min"], bounds["y_max"], self.bev_size, endpoint=False)
        xs += self.cell_size_x / 2
        ys += self.cell_size_y / 2

        num_dirs = 8
        angles = np.linspace(0, 2 * np.pi, num_dirs, endpoint=False)
        ray_dirs_2d = np.stack([np.cos(angles), np.sin(angles), np.zeros(num_dirs)], axis=1)

        for iy in range(self.bev_size):
            origins = []
            directions = []
            for ix in range(self.bev_size):
                z_eye = bounds["z_min"] + (bounds["z_max"] - bounds["z_min"]) * 0.3 + self.PLAYER_HEIGHT
                for d in ray_dirs_2d:
                    origins.append([xs[ix], ys[iy], z_eye])
                    directions.append(d)

            origins = np.array(origins)
            directions = np.array(directions)

            hits = mesh.ray.intersects_any(
                ray_origins=origins,
                ray_directions=directions,
            )
            hits = hits.reshape(-1, num_dirs)
            cover_map[iy] = hits.sum(axis=1) / num_dirs

        return cover_map

    # ----------------------------------------------------------------
    # Cache loading
    # ----------------------------------------------------------------

    def load_cached(self, bev_path: str) -> Tuple[np.ndarray, dict]:
        """Load a pre-computed static BEV and its bounds."""
        bev = np.load(bev_path)
        bounds_path = bev_path.replace('.npy', '_bounds.json')
        with open(bounds_path, 'r') as f:
            bounds = json.load(f)
        self.map_bounds = bounds
        self.cell_size_x = (bounds["x_max"] - bounds["x_min"]) / self.bev_size
        self.cell_size_y = (bounds["y_max"] - bounds["y_min"]) / self.bev_size
        return bev, bounds

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to BEV grid indices."""
        ix = int((x - self.map_bounds["x_min"]) / self.cell_size_x)
        iy = int((y - self.map_bounds["y_min"]) / self.cell_size_y)
        ix = max(0, min(self.bev_size - 1, ix))
        iy = max(0, min(self.bev_size - 1, iy))
        return ix, iy


# ============================================================
# Dynamic BEV Builder
# ============================================================

class DynamicBEVBuilder:
    """
    Constructs per-frame dynamic BEV layers from player poses.

    ONLY uses data available at inference time (positions + orientations
    derived from input poses). No game events, health, or fire actions.

    Dynamic Channels:
      0: Friendly player positions (Gaussian heatmap)
      1: Enemy player positions (Gaussian heatmap)
      2: Current player position + orientation arrow
    """

    def __init__(
        self,
        bev_size: int = 256,
        map_bounds: Optional[dict] = None,
        gaussian_sigma: float = 3.0,
    ):
        self.bev_size = bev_size
        self.map_bounds = None
        self.gaussian_sigma = gaussian_sigma

        # Pre-compute Gaussian kernel
        k = int(gaussian_sigma * 3)
        y, x = np.mgrid[-k:k+1, -k:k+1]
        self.gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * gaussian_sigma**2))
        self.gaussian_kernel /= self.gaussian_kernel.max()
        self.kernel_half = k

        if map_bounds is not None:
            self.set_map_bounds(map_bounds)

    def set_map_bounds(self, bounds: dict):
        self.map_bounds = bounds
        self.cell_size_x = (bounds["x_max"] - bounds["x_min"]) / self.bev_size
        self.cell_size_y = (bounds["y_max"] - bounds["y_min"]) / self.bev_size

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to BEV grid indices."""
        ix = int((x - self.map_bounds["x_min"]) / self.cell_size_x)
        iy = int((y - self.map_bounds["y_min"]) / self.cell_size_y)
        ix = max(0, min(self.bev_size - 1, ix))
        iy = max(0, min(self.bev_size - 1, iy))
        return ix, iy

    def build_frame(
        self,
        current_player_idx: int,
        current_player_team: int,
        all_players: List[Dict],
    ) -> np.ndarray:
        """
        Build dynamic BEV for a single frame.

        Args:
            current_player_idx: Index of the player being generated.
            current_player_team: Team ID of the current player.
            all_players: List of dicts, each containing:
                {"x", "y", "yaw", "team_id", "alive"}

        Returns:
            dynamic_bev: [3, bev_size, bev_size] numpy array
        """
        assert self.map_bounds is not None, "Call set_map_bounds() first"

        bev = np.zeros((NUM_DYNAMIC_CHANNELS, self.bev_size, self.bev_size),
                        dtype=np.float32)

        for i, player in enumerate(all_players):
            if not player.get("alive", True):
                continue

            px, py = player["x"], player["y"]
            ix, iy = self.world_to_grid(px, py)

            if i == current_player_idx:
                # Channel 2: Current player position + orientation
                self._draw_gaussian(bev[2], ix, iy, intensity=1.0)
                self._draw_direction(bev[2], ix, iy, player["yaw"])
            elif player["team_id"] == current_player_team:
                # Channel 0: Friendly positions
                self._draw_gaussian(bev[0], ix, iy, intensity=1.0)
            else:
                # Channel 1: Enemy positions
                self._draw_gaussian(bev[1], ix, iy, intensity=1.0)

        return bev

    def build_sequence(
        self,
        current_player_idx: int,
        current_player_team: int,
        all_players_sequence: List[List[Dict]],
    ) -> np.ndarray:
        """
        Build dynamic BEV for an entire frame sequence.

        Returns:
            dynamic_bev_seq: [num_frames, 3, bev_size, bev_size]
        """
        num_frames = len(all_players_sequence)
        bev_seq = np.zeros(
            (num_frames, NUM_DYNAMIC_CHANNELS, self.bev_size, self.bev_size),
            dtype=np.float32,
        )

        for t in range(num_frames):
            bev_seq[t] = self.build_frame(
                current_player_idx, current_player_team,
                all_players_sequence[t],
            )

        return bev_seq

    def _draw_gaussian(self, channel: np.ndarray, cx: int, cy: int, intensity: float = 1.0):
        """Stamp a Gaussian blob at (cx, cy)."""
        k = self.kernel_half
        y0 = max(0, cy - k)
        y1 = min(self.bev_size, cy + k + 1)
        x0 = max(0, cx - k)
        x1 = min(self.bev_size, cx + k + 1)

        ky0 = y0 - (cy - k)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (cx - k)
        kx1 = kx0 + (x1 - x0)

        channel[y0:y1, x0:x1] = np.maximum(
            channel[y0:y1, x0:x1],
            self.gaussian_kernel[ky0:ky1, kx0:kx1] * intensity,
        )

    def _draw_direction(self, channel: np.ndarray, cx: int, cy: int, yaw_deg: float):
        """Draw a short direction indicator from (cx, cy) along yaw angle."""
        length = self.gaussian_sigma * 2
        yaw_rad = math.radians(yaw_deg)
        dx = math.cos(yaw_rad)
        dy = math.sin(yaw_rad)

        for step in range(int(length)):
            px = int(cx + dx * step)
            py = int(cy + dy * step)
            if 0 <= px < self.bev_size and 0 <= py < self.bev_size:
                channel[py, px] = max(channel[py, px], 0.8)


# ============================================================
# Convenience: estimate bounds from player positions only
# ============================================================

def estimate_bounds_from_positions(
    positions: np.ndarray,
    padding_factor: float = 1.2,
) -> dict:
    """
    Estimate map bounds from player position data when neither mesh nor navmesh
    is available. Ensures minimum size to avoid division by zero.
    """
    x_min, y_min = positions[:, 0].min(), positions[:, 1].min()
    x_max, y_max = positions[:, 0].max(), positions[:, 1].max()

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    rx = (x_max - x_min) / 2 * padding_factor
    ry = (y_max - y_min) / 2 * padding_factor

    # Ensure square aspect ratio with minimum size to avoid division by zero
    r = max(rx, ry, 50.0)

    z_min = positions[:, 2].min() if positions.shape[1] > 2 else 0.0
    z_max = positions[:, 2].max() if positions.shape[1] > 2 else 256.0

    return {
        "x_min": cx - r, "x_max": cx + r,
        "y_min": cy - r, "y_max": cy + r,
        "z_min": float(z_min), "z_max": float(z_max),
    }
