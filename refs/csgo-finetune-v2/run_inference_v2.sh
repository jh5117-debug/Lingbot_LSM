#!/bin/bash
# ==============================================================
# v2 推理：使用单模型(low_noise_model)全参微调checkpoint
# ==============================================================
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
LINGBOT_CODE="/home/nvme02/lingbot-world/code/lingbot-world"
# 微调checkpoint（默认用4ep修复版，按需改成其他epoch）
FT_MODEL_DIR="/home/nvme02/lingbot-world/output/fullft_lowonly_4ep_v2/epoch_4"
# 测试clip（自动选第一个）
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v2"
CLIP_DIR=$(ls -d ${DATASET_DIR}/train/clips/*/ 2>/dev/null | head -1)
if [ -z "${CLIP_DIR}" ]; then
    echo "No clips found in ${DATASET_DIR}/train/clips/"
    exit 1
fi
echo "Using test clip: ${CLIP_DIR}"
IMAGE="${CLIP_DIR}/image.jpg"
PROMPT=$(cat "${CLIP_DIR}/prompt.txt" 2>/dev/null || echo "First-person view of CS:GO competitive gameplay")
torchrun --nproc_per_node=8 inference_csgo.py \
    --ckpt_dir "${CKPT_DIR}" \
    --lingbot_code_dir "${LINGBOT_CODE}" \
    --ft_model_dir "${FT_MODEL_DIR}" \
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
    --save_file "/home/nvme02/lingbot-world/output/inference_v2/latest_output.mp4"
echo ""
echo "Output: /home/nvme02/lingbot-world/output/inference_v2/latest_output.mp4"
