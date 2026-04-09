#!/usr/bin/env python3
"""
Build a static BEV from navmesh.json (standalone CLI script).

Usage:
    python stage2/build_static_bev.py \
        --navmesh /path/to/navmesh.json \
        --output /path/to/static_bev.npy \
        --bev_size 256

Outputs:
    <output>.npy          — [4, H, W] static BEV array
    <output>_bounds.json  — map coordinate bounds
    <output>_place_ids.json — place_name -> normalized ID mapping
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# Allow running from repo root: python stage2/build_static_bev.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2.bev_builder import StaticBEVBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build static BEV from navmesh.json"
    )
    parser.add_argument(
        "--navmesh", required=True,
        help="Path to navmesh.json"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the static BEV (.npy)"
    )
    parser.add_argument(
        "--bev_size", type=int, default=256,
        help="BEV grid resolution (default: 256)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate input
    if not os.path.isfile(args.navmesh):
        logging.error(f"Navmesh file not found: {args.navmesh}")
        sys.exit(1)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Build
    logging.info(f"Building static BEV (size={args.bev_size}) from navmesh...")
    t0 = time.time()

    builder = StaticBEVBuilder(bev_size=args.bev_size)
    bev = builder.build_from_navmesh(args.navmesh, save_path=args.output)

    elapsed = time.time() - t0

    # Summary
    walkable_cells = int(bev[0].sum())
    total_cells = args.bev_size * args.bev_size
    hiding_cells = int((bev[2] > 0).sum())
    has_semantic = int((bev[3] > 0).sum())

    print()
    print("=" * 50)
    print("Static BEV Build Summary")
    print("=" * 50)
    print(f"  Navmesh:        {args.navmesh}")
    print(f"  BEV size:       {args.bev_size}x{args.bev_size}")
    print(f"  Shape:          {bev.shape}")
    print(f"  Walkable:       {walkable_cells}/{total_cells} "
          f"({100*walkable_cells/total_cells:.1f}%)")
    print(f"  Hiding spots:   {hiding_cells} cells painted")
    print(f"  Semantic:       {has_semantic} cells with region labels")
    print(f"  Height range:   [{bev[1].min():.3f}, {bev[1].max():.3f}]")
    print(f"  Time:           {elapsed:.2f}s")
    print(f"  Output:         {args.output}")
    print(f"  Bounds:         {args.output.replace('.npy', '_bounds.json')}")
    print(f"  Place IDs:      {args.output.replace('.npy', '_place_ids.json')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
