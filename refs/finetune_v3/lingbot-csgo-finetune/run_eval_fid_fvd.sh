#!/bin/bash
# ==============================================================
# FID / FVD Evaluation
# ==============================================================
# Run after eval_batch.py has generated videos.
# Compares generated *_gen.mp4 against GT video.mp4 clips.
#
# Dependencies:
#   pip install clean-fid scipy
#   I3D weights auto-downloaded on first run (~330MB)
#
# Output: eval_fid_fvd_report.json with FID and FVD scores.
# ==============================================================

set -e

export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/home/nvme02/tmp

# ============================================================
# Paths — UPDATE THESE
# ============================================================

# Generated videos from eval_batch.py
GEN_DIR="/home/nvme02/lingbot-world/output/eval_stage1/videos"

# Ground-truth clips directory
REAL_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3/val/clips"

# Output directory
OUTPUT_DIR="/home/nvme02/lingbot-world/output/eval_fid_fvd"

# I3D weights (auto-downloaded if missing)
I3D_PATH="/home/nvme02/lingbot-world/models/i3d_torchscript.pt"

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
DEVICE="cuda:0"
FRAME_STRIDE=4     # Sub-sample every 4th frame for FID
CLIP_LEN=16        # 16-frame clips for FVD (standard)
NUM_CLIPS=0        # 0 = use all matched pairs

echo "========================================="
echo "FID / FVD Evaluation"
echo "========================================="
echo "Gen videos:  ${GEN_DIR}"
echo "Real clips:  ${REAL_DIR}"
echo "Output:      ${OUTPUT_DIR}"
echo "I3D weights: ${I3D_PATH}"
echo "Device:      ${DEVICE}"
echo "========================================="

python eval_fid_fvd.py \
    --gen_dir "${GEN_DIR}" \
    --real_dir "${REAL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --i3d_path "${I3D_PATH}" \
    --device "${DEVICE}" \
    --frame_stride ${FRAME_STRIDE} \
    --clip_len ${CLIP_LEN} \
    --num_clips ${NUM_CLIPS}

echo ""
echo "=========================================="
echo "FID/FVD evaluation complete!"
echo "Report: ${OUTPUT_DIR}/eval_fid_fvd_report.json"
echo "=========================================="
