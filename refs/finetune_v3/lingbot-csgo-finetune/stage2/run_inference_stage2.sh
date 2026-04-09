#!/bin/bash
# Stage 2 Multi-Player Inference
# Generates N consistent first-person videos from the same episode

set -e

export TMPDIR=/home/nvme02/tmp

# ============================================================
# Paths — UPDATE THESE
# ============================================================
BASE_MODEL="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
STAGE1_CKPT="/home/nvme02/lingbot-world/output/dual_ft_v3/final"
STAGE2_CKPT="/home/nvme02/lingbot-world/output/stage2_2a/final"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"

# Player clip directories (comma-separated)
# Each directory should contain: video.mp4, poses.npy, action.npy, intrinsics.npy
PROCESSED_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3"
PLAYER_DIRS="${PROCESSED_DIR}/val/clips/clip_0000,${PROCESSED_DIR}/val/clips/clip_0001,${PROCESSED_DIR}/val/clips/clip_0002"

# Optional: static BEV from map geometry (if available from preprocessing)
STATIC_BEV_PATH=""  # e.g., /home/nvme02/lingbot-world/datasets/stage2_data/static_bev.npy

# Output
OUTPUT_DIR="/home/nvme02/lingbot-world/output/stage2_inference"

# Conda
source activate /home/nvme02/envs/lingbot-ft

echo "========================================="
echo "Stage 2 Multi-Player Inference"
echo "========================================="

# ---- Phase 2a inference (BEV-only, no cross-player) ----
# BEV and visibility are built on-the-fly from poses.npy (self-contained).
# To use preprocessed files instead, add: --bev_dir /path/to/stage2_data --episode_id XXXXXX
echo "Running Phase 2a inference (BEV-only, self-contained)..."

STATIC_BEV_ARG=""
if [ -n "$STATIC_BEV_PATH" ]; then
    STATIC_BEV_ARG="--static_bev_path $STATIC_BEV_PATH"
fi

python stage2/inference_stage2.py \
    --base_model_dir "$BASE_MODEL" \
    --stage1_ckpt "$STAGE1_CKPT" \
    --stage2_ckpt "$STAGE2_CKPT" \
    --lingbot_code_dir "$LINGBOT_CODE" \
    --player_dirs "$PLAYER_DIRS" \
    --output_dir "${OUTPUT_DIR}/phase2a" \
    --sampling_steps 70 \
    --guide_scale 5.0 \
    --frame_num 81 \
    $STATIC_BEV_ARG

echo ""
echo "Done! Output: ${OUTPUT_DIR}/phase2a"

# ---- Phase 2b inference (with cross-player attention) ----
# Uncomment after Phase 2b training is complete:
#
# STAGE2B_CKPT="/home/nvme02/lingbot-world/output/stage2_2b/final"
# echo "Running Phase 2b inference (BEV + cross-player, self-contained)..."
# python stage2/inference_stage2.py \
#     --base_model_dir "$BASE_MODEL" \
#     --stage1_ckpt "$STAGE1_CKPT" \
#     --stage2_ckpt "$STAGE2B_CKPT" \
#     --lingbot_code_dir "$LINGBOT_CODE" \
#     --player_dirs "$PLAYER_DIRS" \
#     --output_dir "${OUTPUT_DIR}/phase2b" \
#     --sampling_steps 70 \
#     --guide_scale 5.0 \
#     --frame_num 81 \
#     --enable_cross_player \
#     $STATIC_BEV_ARG
