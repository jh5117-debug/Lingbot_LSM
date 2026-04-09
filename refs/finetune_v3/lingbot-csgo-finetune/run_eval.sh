#!/bin/bash
# ==============================================================
# Batch Evaluation: Val Set Inference + Metrics (PSNR/SSIM/LPIPS)
# ==============================================================
# Runs inference on all val clips and computes quantitative metrics.
# Results: per-clip CSV + aggregate JSON report.
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

# Fine-tuned checkpoint (dual-model, from Stage 1)
FT_CKPT_DIR="/home/nvme02/lingbot-world/output/dual_ft_v3/final"

# Output directory for evaluation results
OUTPUT_DIR="/home/nvme02/lingbot-world/output/eval_stage1"

# Conda
source activate /home/nvme02/envs/lingbot-ft

# ============================================================
# Options (adjust as needed)
# ============================================================
NUM_GPUS=8
SPLIT="val"
MAX_SAMPLES=0          # 0 = all clips, set >0 for quick test
SAMPLE_STEPS=70
GUIDE_SCALE=5.0
FRAME_NUM=81

echo "========================================="
echo "Batch Evaluation — Stage 1"
echo "========================================="
echo "Checkpoint:  ${FT_CKPT_DIR}"
echo "Dataset:     ${DATASET_DIR}"
echo "Split:       ${SPLIT}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPUs:        ${NUM_GPUS}"
echo "Max samples: ${MAX_SAMPLES} (0=all)"
echo "========================================="

torchrun --nproc_per_node=${NUM_GPUS} eval_batch.py \
    --ckpt_dir "${CKPT_DIR}" \
    --lingbot_code_dir "${LINGBOT_CODE}" \
    --ft_ckpt_dir "${FT_CKPT_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --guide_scale "${GUIDE_SCALE}" \
    --frame_num "${FRAME_NUM}" \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size ${NUM_GPUS} \
    --skip_existing

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results: ${OUTPUT_DIR}/eval_report.json"
echo "CSV:     ${OUTPUT_DIR}/metrics.csv"
echo "Videos:  ${OUTPUT_DIR}/videos/"
echo "=========================================="
