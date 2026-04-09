"""
Stage 2 Data Preprocessing: Pre-compute BEV maps and visibility matrices.

This script runs AFTER Stage 1 preprocessing and prepares additional
data needed for Stage 2 training:

1. Static BEV: From 3D mesh (once per map)
2. Dynamic BEV: Per-clip player state overlay
3. Visibility matrices: Per-episode pairwise player visibility

Usage:
    python stage2/preprocess_stage2.py \\
        --raw_data_dir /path/to/raw_csgo_v3/dust2-80-32fps/.../e61.../train \\
        --processed_dir /path/to/processed_csgo_v3 \\
        --output_dir /path/to/stage2_data \\
        --mesh_path /path/to/export/meshes/de_dust2.obj \\
        --static_props_path /path/to/export/static_props.json \\
        --bev_size 256

    # Without mesh (position-only BEV + FOV-only visibility):
    python stage2/preprocess_stage2.py \\
        --raw_data_dir /path/to/raw_data/train \\
        --processed_dir /path/to/processed_csgo_v3 \\
        --output_dir /path/to/stage2_data \\
        --bev_size 256 --no_mesh
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2.bev_builder import StaticBEVBuilder, DynamicBEVBuilder, estimate_bounds_from_positions
from stage2.visibility import VisibilityComputer, precompute_visibility


def find_episodes(raw_data_dir):
    """Find all episode directories and their player action files."""
    episodes = {}

    for ep_name in sorted(os.listdir(raw_data_dir)):
        ep_dir = os.path.join(raw_data_dir, ep_name)
        if not os.path.isdir(ep_dir) or not ep_name.startswith("Ep_"):
            continue

        episode_id = ep_name.replace("Ep_", "")
        players = []

        for fname in sorted(os.listdir(ep_dir)):
            if fname.endswith(".json") and not fname.endswith("_episode_info.json") \
                    and not fname.endswith("_video_manifest.json") \
                    and fname != "game_manifest.json" \
                    and fname != "world_events.jsonl":
                # This is a player action file
                stem = fname.replace(".json", "")
                m = re.search(r'team_(\d+)_player_(\d+)', stem)
                if m:
                    players.append({
                        "stem": stem,
                        "action_path": os.path.join(ep_dir, fname),
                        "team_id": int(m.group(1)),
                        "player_idx": int(m.group(2)),
                    })

        if players:
            # Check for world events
            events_path = os.path.join(ep_dir, "world_events.jsonl")
            manifest_path = os.path.join(ep_dir, "game_manifest.json")

            episodes[episode_id] = {
                "dir": ep_dir,
                "players": sorted(players, key=lambda p: p["player_idx"]),
                "events_path": events_path if os.path.exists(events_path) else None,
                "manifest_path": manifest_path if os.path.exists(manifest_path) else None,
            }

    return episodes


def load_player_positions_from_actions(action_path, subsample=4):
    """
    Load player positions/rotations from action JSON.
    Returns list of dicts with x, y, z, yaw, health, fire, alive.
    """
    with open(action_path, 'r') as f:
        frames = json.load(f)

    positions = []
    for i in range(0, len(frames), subsample):
        frame = frames[i]

        # Position priority: render_transform > camera_position > root
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

        # Rotation
        cam_rot = frame.get("camera_rotation")
        yaw = cam_rot[2] if cam_rot else frame.get("yaw", 0)

        # Health and actions
        health = frame.get("health", 100)
        action = frame.get("action", {})
        fire = action.get("fire", False) if isinstance(action, dict) else False

        positions.append({
            "x": x, "y": y, "z": z,
            "yaw": yaw,
            "health": health,
            "fire": fire,
            "alive": health > 0,
        })

    return positions


def load_world_events(events_path, subsample=4):
    """
    Load world events and index them by frame.
    Returns dict: frame_idx -> list of active events.
    """
    events_by_frame = defaultdict(list)

    if not events_path or not os.path.exists(events_path):
        return events_by_frame

    with open(events_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            frame = event.get("frame_count", 0) // subsample

            etype = event.get("event_type", "")
            data = event.get("data", {})

            if etype in ("flashbang_detonate", "smokegrenade_detonate",
                         "hegrenade_detonate"):
                events_by_frame[frame].append({
                    "type": "flash" if "flash" in etype else "smoke",
                    "x": data.get("x", 0),
                    "y": data.get("y", 0),
                    "radius": 15.0,  # grid cells
                })
            elif etype == "inferno_startburn":
                # Fire persists for ~7 seconds (about 112 frames at 16fps)
                for dt in range(112 // subsample):
                    events_by_frame[frame + dt].append({
                        "type": "fire",
                        "x": data.get("x", 0),
                        "y": data.get("y", 0),
                        "radius": 10.0,
                    })

    return events_by_frame


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Data Preprocessing")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                        help="Raw data train/ directory containing Ep_* subdirs")
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Stage 1 processed data directory (with metadata CSV)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Stage 2 output directory")
    parser.add_argument("--mesh_path", type=str, default="",
                        help="Path to map .obj mesh file")
    parser.add_argument("--static_props_path", type=str, default="",
                        help="Path to static_props.json")
    parser.add_argument("--bev_size", type=int, default=256)
    parser.add_argument("--subsample", type=int, default=4,
                        help="Frame subsample rate for BEV/visibility")
    parser.add_argument("--no_mesh", action="store_true",
                        help="Skip mesh-based computations")
    parser.add_argument("--skip_static_bev", action="store_true",
                        help="Skip static BEV if already computed")
    parser.add_argument("--skip_visibility", action="store_true",
                        help="Skip visibility computation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Step 1: Build static BEV
    # ============================================================
    static_bev_path = os.path.join(args.output_dir, "static_bev.npy")

    if args.skip_static_bev and os.path.exists(static_bev_path):
        logging.info("Loading existing static BEV...")
        static_builder = StaticBEVBuilder(bev_size=args.bev_size)
        static_bev, bev_bounds = static_builder.load_cached(static_bev_path)
    elif args.mesh_path and not args.no_mesh:
        logging.info("Building static BEV from mesh...")
        static_builder = StaticBEVBuilder(bev_size=args.bev_size)
        static_bev = static_builder.build_from_mesh(
            map_mesh_path=args.mesh_path,
            static_props_path=args.static_props_path if args.static_props_path else None,
            save_path=static_bev_path,
        )
        bev_bounds = static_builder.map_bounds
    else:
        logging.info("No mesh available. Will estimate bounds from player positions.")
        static_builder = None
        bev_bounds = None
        static_bev = None

    # ============================================================
    # Step 2: Discover episodes
    # ============================================================
    episodes = find_episodes(args.raw_data_dir)
    logging.info(f"Found {len(episodes)} episodes")

    # If no mesh, estimate bounds from all player positions
    if bev_bounds is None:
        logging.info("Estimating map bounds from player positions...")
        all_positions = []
        for ep_id, ep in episodes.items():
            for player in ep["players"][:2]:  # sample a couple players
                positions = load_player_positions_from_actions(
                    player["action_path"], subsample=32
                )
                for p in positions:
                    all_positions.append([p["x"], p["y"], p["z"]])

        if all_positions:
            all_positions = np.array(all_positions)
            bev_bounds = estimate_bounds_from_positions(all_positions)
            logging.info(f"Estimated bounds: {bev_bounds}")

            # Save bounds
            with open(os.path.join(args.output_dir, "static_bev_bounds.json"), 'w') as f:
                json.dump(bev_bounds, f)

            # Create zero static BEV
            static_bev = np.zeros((4, args.bev_size, args.bev_size), dtype=np.float32)
            np.save(static_bev_path, static_bev)
        else:
            logging.error("No player positions found!")
            return

    # ============================================================
    # Step 3: Per-episode processing (dynamic BEV + visibility)
    # ============================================================
    dyn_builder = DynamicBEVBuilder(bev_size=args.bev_size)
    dyn_builder.set_map_bounds(bev_bounds)

    vis_computer = VisibilityComputer(
        fov_degrees=90.0,
        mesh_path=args.mesh_path if (args.mesh_path and not args.no_mesh) else None,
    )

    episodes_dir = os.path.join(args.output_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)

    for ep_id, ep in episodes.items():
        ep_out_dir = os.path.join(episodes_dir, f"Ep_{ep_id}")
        os.makedirs(ep_out_dir, exist_ok=True)

        logging.info(f"Processing episode {ep_id} ({len(ep['players'])} players)...")

        # Load all player positions
        all_player_positions = {}
        for player in ep["players"]:
            positions = load_player_positions_from_actions(
                player["action_path"], subsample=args.subsample
            )
            all_player_positions[player["player_idx"]] = positions

        # Determine number of frames (min across players)
        num_frames = min(len(v) for v in all_player_positions.values())
        player_indices = sorted(all_player_positions.keys())
        num_players = len(player_indices)

        # --- Visibility matrices ---
        if not args.skip_visibility:
            vis_path = os.path.join(ep_out_dir, "visibility.npy")
            if not os.path.exists(vis_path):
                logging.info(f"  Computing visibility ({num_frames} frames, {num_players} players)...")
                vis_matrices = np.zeros((num_frames, num_players, num_players), dtype=np.float32)

                for t in range(num_frames):
                    players_t = []
                    for pi in player_indices:
                        players_t.append(all_player_positions[pi][t])
                    vis_matrices[t] = vis_computer.compute_frame(players_t)

                np.save(vis_path, vis_matrices)
                logging.info(f"  Saved visibility -> {vis_path}")
            else:
                logging.info(f"  Visibility already exists, skipping")

        # --- Dynamic BEV per player ---
        world_events = load_world_events(ep.get("events_path"), subsample=args.subsample)

        for player in ep["players"]:
            pidx = player["player_idx"]
            team_id = player["team_id"]
            stem = player["stem"]

            dyn_bev_path = os.path.join(ep_out_dir, f"dynamic_bev_{stem}.npy")
            if os.path.exists(dyn_bev_path):
                continue

            logging.info(f"  Building dynamic BEV for {stem}...")

            # Build per-frame dynamic BEV
            dyn_bevs = []
            for t in range(num_frames):
                all_players_t = []
                for pi in player_indices:
                    p = all_player_positions[pi][t]
                    p["team_id"] = next(
                        pl["team_id"] for pl in ep["players"] if pl["player_idx"] == pi
                    )
                    all_players_t.append(p)

                # Find index of current player in sorted list
                current_idx_in_list = player_indices.index(pidx)

                events_t = world_events.get(t, None)
                frame_bev = dyn_builder.build_frame(
                    current_player_idx=current_idx_in_list,
                    current_player_team=team_id,
                    all_players=all_players_t,
                    world_events=events_t,
                )
                dyn_bevs.append(frame_bev)

            # Save as time-averaged BEV (reduce storage)
            # For training, we use the mean across all frames in the clip
            dyn_bev_array = np.stack(dyn_bevs, axis=0)  # [T, 6, H, W]

            # Save full sequence (can be sliced per clip later)
            np.save(dyn_bev_path, dyn_bev_array.astype(np.float16))
            logging.info(f"    Saved {dyn_bev_path} shape={dyn_bev_array.shape}")

        # --- Save player index mapping ---
        mapping_path = os.path.join(ep_out_dir, "player_mapping.json")
        if not os.path.exists(mapping_path):
            mapping = {
                "player_indices": player_indices,
                "players": [
                    {"idx": p["player_idx"], "team_id": p["team_id"], "stem": p["stem"]}
                    for p in ep["players"]
                ],
                "num_frames": num_frames,
                "subsample": args.subsample,
            }
            with open(mapping_path, 'w') as f:
                json.dump(mapping, f, indent=2)

    logging.info("Stage 2 preprocessing complete!")


if __name__ == "__main__":
    main()
