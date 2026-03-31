#!/bin/bash
# ==============================================================
# Step 3: Inference with fine-tuned model
# ==============================================================
# Generate CSGO gameplay video from image + action controls
# ==============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"

# ---- Choose one: LoRA or Full FT ----
# Option A: LoRA inference
LORA_PATH="/home/nvme02/lingbot-world/output/csgo_lora_r32/final/lora_weights.pth"

# Option B: Full FT inference (comment out LORA_PATH and uncomment FT_MODEL_DIR)
# FT_MODEL_DIR="/home/nvme02/lingbot-world/output/csgo_full_ft/final"

# ---- Test clip (pick one from val set) ----
# Check metadata_val.csv to find clip paths
TEST_CLIP="/home/nvme02/lingbot-world/datasets/csgo_processed/val/clips"
# Pick the first clip directory
CLIP_DIR=$(ls -d ${TEST_CLIP}/*/ 2>/dev/null | head -1)

if [ -z "${CLIP_DIR}" ]; then
    echo "No validation clips found! Run preprocessing first."
    exit 1
fi

echo "Using test clip: ${CLIP_DIR}"

IMAGE="${CLIP_DIR}/image.jpg"
PROMPT=$(cat "${CLIP_DIR}/prompt.txt" 2>/dev/null || echo "First-person view of CS:GO competitive gameplay")

# ---- Run inference ----
torchrun --nproc_per_node=8 inference_csgo.py \
    --ckpt_dir "${CKPT_DIR}" \
    --lingbot_code_dir "${LINGBOT_CODE}" \
    --lora_path "${LORA_PATH}" \
    --image "${IMAGE}" \
    --action_path "${CLIP_DIR}" \
    --prompt "${PROMPT}" \
    --size "480*832" \
    --frame_num 81 \
    --sample_steps 70 \
    --sample_shift 10.0 \
    --guide_scale 5.0 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \
    --save_file "csgo_inference_output.mp4"

echo ""
echo "=========================================="
echo "Inference complete!"
echo "Output: csgo_inference_output.mp4"
echo "=========================================="
