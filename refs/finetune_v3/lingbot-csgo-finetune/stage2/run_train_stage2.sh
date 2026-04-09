#!/bin/bash
# Stage 2 Training: Multi-View CSGO Generation
# Dual-model alternating training with BEV + cross-player attention

set -e

export TMPDIR=/home/nvme02/tmp

# ============================================================
# Paths — UPDATE THESE
# ============================================================
STAGE1_CKPT="/home/nvme02/lingbot-world/output/dual_ft_v3/epoch_2"
BASE_MODEL_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3"
STAGE2_DATA="/home/nvme02/lingbot-world/datasets/stage2_data"

# Output
PHASE="${1:-2a}"  # Pass "2a" or "2b" as first argument
OUTPUT_DIR="/home/nvme02/lingbot-world/output/stage2_${PHASE}"

# Phase 2b: needs Phase 2a checkpoint
STAGE2A_CKPT=""
if [ "$PHASE" = "2b" ]; then
    STAGE2A_CKPT="/home/nvme02/lingbot-world/output/stage2_2a/final"
fi

# Conda (skip if already activated)
source activate /home/nvme02/envs/lingbot-ft 2>/dev/null || true
export PATH=/home/nvme02/envs/lingbot-ft/bin:$PATH

echo "========================================="
echo "Stage 2 Training — Phase ${PHASE}"
echo "========================================="
echo "Stage 1 checkpoint: $STAGE1_CKPT"
echo "Output: $OUTPUT_DIR"
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

# Launch training
accelerate launch \
    --config_file accelerate_config_dual.yaml \
    stage2/train_stage2.py \
    --ckpt_dir "$STAGE1_CKPT" \
    --base_model_dir "$BASE_MODEL_DIR" \
    --lingbot_code_dir "$LINGBOT_CODE" \
    --dataset_dir "$DATASET_DIR" \
    --stage2_dir "$STAGE2_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --phase "$PHASE" \
    --stage2_ckpt "$STAGE2A_CKPT" \
    --num_epochs 6 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --save_every_n_epochs 1 \
    --gradient_checkpointing \
    --bev_channels 7 \
    --bev_size 256 \
    --bev_token_grid 16 \
    --num_context_players 2 \
    --context_noise_std 0.1 \
    --gate_reg_weight 0.01

echo ""
echo "Training complete! Checkpoints saved to: $OUTPUT_DIR"
