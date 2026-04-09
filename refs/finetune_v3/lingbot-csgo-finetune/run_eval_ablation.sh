#!/bin/bash
# ==============================================================
# Ablation Study Evaluation — Stage 1
# Runs PSNR/SSIM/LPIPS evaluation for all ablation variants:
#   - Zero-shot baseline (no fine-tuning)
#   - Per-epoch checkpoints (convergence curve)
#   - Final dual-model (ours)
#   - LoRA fine-tune (if available)
#   - Single-model ablation (if available)
# ==============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/nvme02/tmp

# ============================================================
# Paths — UPDATE THESE
# ============================================================
BASE_MODEL="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3"
DUAL_FT_DIR="/home/nvme02/lingbot-world/output/dual_ft_v3"
OUTPUT_BASE="/home/nvme02/lingbot-world/output/ablation"

source activate /home/nvme02/envs/lingbot-ft

# Number of clips per ablation (50 is enough for ablation, saves time)
MAX_SAMPLES=50

# ============================================================
# Helper function
# ============================================================
run_eval() {
    local name="$1"
    local ft_args="$2"
    local out_dir="${OUTPUT_BASE}/${name}"

    echo ""
    echo "============================================"
    echo "Evaluating: ${name}"
    echo "Output:     ${out_dir}"
    echo "============================================"

    torchrun --nproc_per_node=8 eval_batch.py \
        --ckpt_dir "${BASE_MODEL}" \
        --lingbot_code_dir "${LINGBOT_CODE}" \
        ${ft_args} \
        --dataset_dir "${DATASET_DIR}" \
        --output_dir "${out_dir}" \
        --split val \
        --max_samples ${MAX_SAMPLES} \
        --sample_steps 70 \
        --guide_scale 5.0 \
        --frame_num 81 \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 8 \
        --skip_existing

    echo "Done: ${name} → ${out_dir}/eval_report.json"
}

mkdir -p "${OUTPUT_BASE}"

# ============================================================
# 1. Zero-shot baseline (base model, no fine-tuning)
# ============================================================
run_eval "zeroshot" ""

# ============================================================
# 2. Per-epoch convergence curve
# ============================================================
for epoch in 2 4 6 8 10; do
    CKPT="${DUAL_FT_DIR}/epoch_${epoch}"
    if [ -d "${CKPT}" ]; then
        run_eval "epoch_${epoch}" "--ft_ckpt_dir ${CKPT}"
    else
        echo "[SKIP] epoch_${epoch}: checkpoint not found at ${CKPT}"
    fi
done

# ============================================================
# 3. Final checkpoint (our full method)
# ============================================================
FINAL_CKPT="${DUAL_FT_DIR}/final"
if [ -d "${FINAL_CKPT}" ]; then
    run_eval "final_dual_model" "--ft_ckpt_dir ${FINAL_CKPT}"
else
    echo "[SKIP] final: checkpoint not found at ${FINAL_CKPT}"
fi

# ============================================================
# 4. LoRA ablation (if trained)
# ============================================================
LORA_PATH="/home/nvme02/lingbot-world/output/lora_ft/final/lora_weights.pth"
if [ -f "${LORA_PATH}" ]; then
    run_eval "lora_finetune" "--lora_path ${LORA_PATH}"
else
    echo "[SKIP] LoRA: checkpoint not found at ${LORA_PATH}"
fi

# ============================================================
# 5. Single-model ablation (if trained)
# ============================================================
SINGLE_MODEL_CKPT="/home/nvme02/lingbot-world/output/single_model_ft/final"
if [ -d "${SINGLE_MODEL_CKPT}" ]; then
    run_eval "single_model" "--ft_ckpt_dir ${SINGLE_MODEL_CKPT}"
else
    echo "[SKIP] single_model: checkpoint not found at ${SINGLE_MODEL_CKPT}"
fi

# ============================================================
# 6. Aggregate all ablation results into a comparison table
# ============================================================
echo ""
echo "============================================"
echo "Aggregating ablation results..."
echo "============================================"
python3 aggregate_ablation.py --ablation_dir "${OUTPUT_BASE}"

echo ""
echo "============================================"
echo "Ablation evaluation complete!"
echo "Summary: ${OUTPUT_BASE}/ablation_summary.csv"
echo "============================================"
