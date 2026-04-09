#!/bin/bash
# ==============================================================
# Step 1: Preprocess CSGO dataset
# ==============================================================
# Run this FIRST on the server to convert raw CSGO data → training clips
#
# Input:  /home/nvme02/lingbot-world/datasets/datasets - poc/
# Output: /home/nvme02/lingbot-world/datasets/csgo_processed/
# ==============================================================

set -e

INPUT_DIR="/home/nvme02/lingbot-world/datasets/datasets - poc"
OUTPUT_DIR="/home/nvme02/lingbot-world/datasets/csgo_processed"

# Validation episodes: 2 per map (1 Nuke + 1 Mirage from different matches)
# Adjust these patterns to match actual episode IDs in your dataset.
# After running once, check the metadata_val.csv to verify the split.
VAL_EPISODES="thunderflash_vs_wu_tang_m1_mirage_0000_000001,virtus_pro_vs_9z_m1_mirage_0000_000002,evil_geniuses_vs_kari_m1_nuke_0000_000001,aurora_vs_falcons_m1_nuke_0000_000003"  # Will be set after first run when we know exact episode IDs

python preprocess_csgo.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clip_frames 81 \
    --target_fps 16 \
    --height 480 \
    --width 832 \
    --stride 40 \
    --val_episodes "${VAL_EPISODES}"

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "Check output at: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Review metadata_train.csv and metadata_val.csv"
echo "2. Set VAL_EPISODES in this script and re-run if needed"
echo "3. Run: bash run_train_lora.sh"
echo "=========================================="
