#!/bin/bash
# ==============================================================
# Dual-Model MoE Fine-tuning
# ==============================================================
# Trains both high_noise_model and low_noise_model on CSGO data.
# Even epochs: low_noise_model (t < 947)
# Odd epochs:  high_noise_model (t >= 947)
#
# Hardware: 8 × H20 96GB (both models need more VRAM)
# Expected: ~60-80GB VRAM per GPU, ~4-6 hours for 10 epochs
# ==============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v2"
OUTPUT_DIR="/home/nvme02/lingbot-world/output/dual_ft_v3"

accelerate launch \
    --config_file accelerate_config_dual.yaml \
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
    --num_epochs 10 \
    --gradient_accumulation_steps 4 \
    --save_every_n_epochs 2 \
    --dataset_repeat 1 \
    --gradient_checkpointing

echo ""
echo "=========================================="
echo "Dual-model training complete!"
echo "Checkpoints at: ${OUTPUT_DIR}"
echo "Each checkpoint contains both low_noise_model/ and high_noise_model/"
echo "=========================================="
