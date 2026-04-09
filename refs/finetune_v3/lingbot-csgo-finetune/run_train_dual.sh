#!/bin/bash
# ==============================================================
# Dual-Model MoE Fine-tuning (Sequential)
# ==============================================================
# Trains low_noise_model (t < 947) then high_noise_model (t >= 947).
# Each model runs as a separate process with its own DeepSpeed engine.
#
# Speed: ~8-10x faster than old CPU-offload version
#   Old: ~16 min/step (CPU offload + dual engine)
#   New: ~1-2 min/step (ZeRO-3 no offload + single engine)
#
# Hardware: 8 × H20 96GB
# Expected: ~4-7 hours per model (5 epochs), ~10-14 hours total
# ==============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/nvme02/tmp

# ============================================================
# Paths — UPDATE THESE
# ============================================================
CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3"
OUTPUT_DIR="/home/nvme02/lingbot-world/output/dual_ft_v3"
CODE_DIR="/home/nvme02/lingbot-world/code/finetune_v3/lingbot-csgo-finetune"

# Conda
source activate /home/nvme02/envs/lingbot-ft

# ============================================================
# Training config
# ============================================================
EPOCHS_PER_MODEL=5      # 5 epochs each = 10 total
LR=1e-5
GRAD_ACCUM=4
SAVE_EVERY=1            # Save checkpoint every epoch

cd "${CODE_DIR}"

# ============================================================
# Phase 1: Train low_noise_model (handles t < 947, texture/detail)
# ============================================================
echo ""
echo "=========================================="
echo "Phase 1: Training low_noise_model"
echo "  Epochs: ${EPOCHS_PER_MODEL}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

accelerate launch \
    --config_file accelerate_config_dual.yaml \
    train_lingbot_csgo.py \
    --model_type low \
    --ckpt_dir "${CKPT_DIR}" \
    --lingbot_code_dir "${LINGBOT_CODE}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs ${EPOCHS_PER_MODEL} \
    --learning_rate ${LR} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_every_n_epochs ${SAVE_EVERY} \
    --gradient_checkpointing \
    2>&1 | tee "${OUTPUT_DIR}/train_low.log"

echo ""
echo "=========================================="
echo "Phase 1 complete: low_noise_model trained"
echo "=========================================="

# ============================================================
# Phase 2: Train high_noise_model (handles t >= 947, structure)
# ============================================================
echo ""
echo "=========================================="
echo "Phase 2: Training high_noise_model"
echo "  Epochs: ${EPOCHS_PER_MODEL}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

accelerate launch \
    --config_file accelerate_config_dual.yaml \
    train_lingbot_csgo.py \
    --model_type high \
    --ckpt_dir "${CKPT_DIR}" \
    --lingbot_code_dir "${LINGBOT_CODE}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs ${EPOCHS_PER_MODEL} \
    --learning_rate ${LR} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_every_n_epochs ${SAVE_EVERY} \
    --gradient_checkpointing \
    2>&1 | tee "${OUTPUT_DIR}/train_high.log"

echo ""
echo "=========================================="
echo "All training complete!"
echo ""
echo "Checkpoints:"
echo "  low_noise_model:  ${OUTPUT_DIR}/final/low_noise_model/"
echo "  high_noise_model: ${OUTPUT_DIR}/final/high_noise_model/"
echo ""
echo "For inference, use --ft_ckpt_dir ${OUTPUT_DIR}/final"
echo "=========================================="
