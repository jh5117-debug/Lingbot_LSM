#!/bin/bash
# ==============================================================
# [DEPRECATED] Use run_train_dual.sh instead
# ==============================================================
# This script was for single low_noise_model training.
# train_lingbot_csgo.py now trains both models (dual MoE).
# ==============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/csgo_processed"
OUTPUT_DIR="/home/nvme02/lingbot-world/output/csgo_full_ft"

accelerate launch \
    --config_file accelerate_config_zero2.yaml \
    train_lingbot_csgo.py \
    --ckpt_dir "${CKPT_DIR}" \
    --lingbot_code_dir "${LINGBOT_CODE}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --num_epochs 50 \
    --gradient_accumulation_steps 4 \
    --save_every_n_epochs 10 \
    --lora_rank 0 \
    --dataset_repeat 1 \
    --gradient_checkpointing

echo ""
echo "=========================================="
echo "Full fine-tuning complete!"
echo "Checkpoints at: ${OUTPUT_DIR}"
echo "=========================================="
