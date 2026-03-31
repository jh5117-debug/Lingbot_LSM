#!/bin/bash
# ==============================================================
# Step 2a: LoRA Fine-tuning (recommended first)
# ==============================================================
# Fine-tune LingBot-World act model on CSGO with LoRA
# Hardware: 8 × H20 96GB
# Expected: ~40GB VRAM per GPU, ~2-4 hours for 50 epochs
# ==============================================================

set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/csgo_processed_v3"
OUTPUT_DIR="/home/nvme02/lingbot-world/output/csgo_fullft_v3"

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
    --num_epochs 10 \
    --gradient_accumulation_steps 4 \
    --save_every_n_epochs 1 \
    --lora_rank 0 \
    --dataset_repeat 1 \
    --gradient_checkpointing

echo ""
echo "=========================================="
echo "LoRA training complete!"
echo "Checkpoints at: ${OUTPUT_DIR}"
echo ""
echo "To test inference, run: bash run_inference.sh"
echo "=========================================="
