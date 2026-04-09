#!/bin/bash
# Stage 2 Preprocessing: BEV + Visibility
# Run on server after Stage 1 preprocessing is complete.

set -e

export TMPDIR=/home/nvme02/tmp

# Paths
RAW_DATA="/home/nvme02/lingbot-world/datasets/raw_csgo_v3/dust2-80-32fps/aef9560bbce0c405/e61e20c503eb4af78d4b2011f945aca0/train"
PROCESSED_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3"
OUTPUT_DIR="/home/nvme02/lingbot-world/datasets/stage2_data"

# Navmesh path (preferred — fast, no trimesh needed)
NAVMESH_PATH="/home/nvme02/lingbot-world/datasets/new_sample/bcc3a2ac490c4888bcb97c3553a25671/train/Ep_000005/navmesh.json"

# Mesh paths (fallback when navmesh is unavailable)
MESH_PATH=""           # e.g., /path/to/export/meshes/de_dust2.obj
STATIC_PROPS_PATH=""   # e.g., /path/to/export/static_props.json

# Static BEV settings
BEV_SIZE=256
STATIC_BEV_PATH="${OUTPUT_DIR}/static_bev.npy"

# Conda
source activate /home/nvme02/envs/lingbot-ft

echo "========================================="
echo "Stage 2 Preprocessing"
echo "========================================="

# --------------------------------------------------
# Step 0: Build static BEV from navmesh (if available)
# --------------------------------------------------
if [ -n "$NAVMESH_PATH" ] && [ -f "$NAVMESH_PATH" ]; then
    echo "Building static BEV from navmesh: $NAVMESH_PATH"
    mkdir -p "$OUTPUT_DIR"
    python stage2/build_static_bev.py \
        --navmesh "$NAVMESH_PATH" \
        --output "$STATIC_BEV_PATH" \
        --bev_size "$BEV_SIZE"
    echo "Static BEV saved to: $STATIC_BEV_PATH"
elif [ -n "$NAVMESH_PATH" ]; then
    echo "WARNING: NAVMESH_PATH set but file not found: $NAVMESH_PATH"
    echo "Skipping static BEV build."
else
    echo "No navmesh path configured, skipping static BEV build."
fi

# --------------------------------------------------
# Step 1: Main preprocessing (per-frame BEV stacking)
# --------------------------------------------------
if [ -z "$MESH_PATH" ]; then
    echo "No mesh provided, using position-only mode"
    python stage2/preprocess_stage2.py \
        --raw_data_dir "$RAW_DATA" \
        --processed_dir "$PROCESSED_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --bev_size "$BEV_SIZE" \
        --subsample 4 \
        --no_mesh
else
    echo "Using mesh: $MESH_PATH"
    python stage2/preprocess_stage2.py \
        --raw_data_dir "$RAW_DATA" \
        --processed_dir "$PROCESSED_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --mesh_path "$MESH_PATH" \
        --static_props_path "$STATIC_PROPS_PATH" \
        --bev_size "$BEV_SIZE" \
        --subsample 4
fi

echo "Done! Output: $OUTPUT_DIR"
