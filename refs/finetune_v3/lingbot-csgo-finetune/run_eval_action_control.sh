#!/bin/bash
# ==============================================================
# Action Controllability Evaluation
# ==============================================================
# Run after eval_batch.py has generated videos.
# Measures whether generated videos follow input actions by
# comparing optical flow against pose/action ground truth.
#
# CPU-only (OpenCV Farneback optical flow). No GPU needed.
#
# Dependencies:
#   pip install opencv-python numpy tqdm
#
# Output: eval_action_control_report.json with per-clip and
#         aggregate action controllability metrics.
# ==============================================================

set -e

export TMPDIR=/home/nvme02/tmp

# ============================================================
# Paths — UPDATE THESE
# ============================================================

# Generated videos from eval_batch.py
GEN_DIR="/home/nvme02/lingbot-world/output/eval_stage1/videos"

# Ground-truth clips directory (with poses.npy, action.npy)
CLIP_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3/val/clips"

# Output directory
OUTPUT_DIR="/home/nvme02/lingbot-world/output/eval_action_control"

# Code directory
CODE_DIR="/home/nvme02/lingbot-world/code/finetune_v3/lingbot-csgo-finetune"

# ============================================================
# Conda
# ============================================================
source activate /home/nvme02/envs/lingbot-ft

cd "${CODE_DIR}"

# ============================================================
# Options
# ============================================================
MAX_SAMPLES=0   # 0 = evaluate all matched clips

echo "========================================="
echo "Action Controllability Evaluation"
echo "========================================="
echo "Gen videos:  ${GEN_DIR}"
echo "Clip dir:    ${CLIP_DIR}"
echo "Output:      ${OUTPUT_DIR}"
echo "Max samples: ${MAX_SAMPLES}"
echo "========================================="

python eval_action_control.py \
    --gen_dir "${GEN_DIR}" \
    --clip_dir "${CLIP_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples ${MAX_SAMPLES}

echo ""
echo "=========================================="
echo "Action controllability evaluation complete!"
echo "Report: ${OUTPUT_DIR}/eval_action_control_report.json"
echo "=========================================="
