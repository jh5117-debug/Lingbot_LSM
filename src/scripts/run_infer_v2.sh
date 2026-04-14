#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_infer_v2.sh
# ============================================================
CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"   # 基础模型目录（必填）
IMAGE="/home/nvme02/Memory-world/data/csgo_val_clips_action4/val/clips/Ep_000028_team_2_player_0001_inst_000_clip0005/image.jpg"
ACTION_PATH="/home/nvme02/Memory-world/data/csgo_val_clips_action4/val/clips/Ep_000028_team_2_player_0001_inst_000_clip0005"
PROMPT="First-person CS:GO competitive gameplay"

# 微调权重（三选一，留空则跑 baseline）
LORA_PATH=""             # LoRA 权重路径（lora_weights.pth）
FT_MODEL_DIR="/home/nvme02/wlx/Memory/outputs/train/v2_stage1_dual/low_noise_model/epoch_1"
                         # 全参微调 / dual-low  目录（如 .../train_v2_stage1_dual/low_noise_model/epoch_2）
FT_HIGH_MODEL_DIR="/home/nvme02/wlx/Memory/outputs/train/v2_stage1_dual/high_noise_model/epoch_1"
                         # dual-high 目录（如 .../train_v2_stage1_dual/high_noise_model/epoch_2）
                         # FT_HIGH_MODEL_DIR 有值时视为 dual 模式，此时 FT_MODEL_DIR 也必须填写

# Memory Bank（训练出的模型设为 true，baseline 保持 false）
USE_MEMORY=true
MEMORY_MAX_SIZE=50

# 推理参数
FRAME_NUM=81             # 单 clip 帧数（81 帧 @ 16fps ≈ 5 秒）
NUM_CLIPS=1              # 当前固定 1 个 clip
                         # TODO 多 clip 推理：将 NUM_CLIPS 改为目标数量（如 5），
                         # 并确保 ACTION_PATH 目录内 action.npy 帧数 >= FRAME_NUM * NUM_CLIPS，
                         # 同时将 USE_MEMORY=true 以利用 Memory Bank 跨 clip 传递上下文。
SAMPLE_STEPS=40
SAMPLE_SHIFT=10.0
GUIDE_SCALE=5.0
SIZE="480*832"

OUTPUT_BASE="/home/nvme02/wlx/Memory/outputs"   # 推理结果根目录
CUDA_VISIBLE_DEVICES="0,1,2,3"

# 设备分配（两个 14B 模型 + VAE/T5，需要至少 3 张卡）
LOW_MODEL_DEVICE="cuda:0"
HIGH_MODEL_DEVICE="cuda:1"
VAE_DEVICE="cuda:2"
T5_DEVICE="cuda:2"
CONVERT_HIGH_TO_MEMORY=true    # 将 high_noise_model 转为 WanModelWithMemory（dual 模式必须为 true）

# ============================================================
# 以下内容通常无需修改
# ============================================================

export CUDA_VISIBLE_DEVICES

# ---------- 路径检查 ----------
_err=0
if [ -z "${CKPT_DIR}" ];    then echo "[ERROR] CKPT_DIR 未设置"    >&2; _err=1; fi
if [ -z "${IMAGE}" ];       then echo "[ERROR] IMAGE 未设置"        >&2; _err=1; fi
if [ -z "${ACTION_PATH}" ]; then echo "[ERROR] ACTION_PATH 未设置"  >&2; _err=1; fi
# dual 模式：FT_HIGH_MODEL_DIR 非空时 FT_MODEL_DIR 也必须填写
if [ -n "${FT_HIGH_MODEL_DIR}" ] && [ -z "${FT_MODEL_DIR}" ]; then
    echo "[ERROR] dual 模式下 FT_MODEL_DIR 未设置（FT_HIGH_MODEL_DIR 有值时必须同时填写 FT_MODEL_DIR）" >&2; _err=1
fi
# 使用训练权重但未启用 Memory Bank 时给出提示
if { [ -n "${FT_MODEL_DIR}" ] || [ -n "${FT_HIGH_MODEL_DIR}" ] || [ -n "${LORA_PATH}" ]; } \
        && [ "${USE_MEMORY}" != "true" ]; then
    echo "[WARN] 检测到微调权重但 USE_MEMORY=false，Memory Bank 未启用；如需使用请将 USE_MEMORY 设为 true" >&2
fi
if [ "${_err}" -ne 0 ]; then exit 1; fi

# ---------- 路径计算 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------- 自动推导实验名 → 输出目录 ----------
# 规则：
#   dual  模式（FT_HIGH_MODEL_DIR 非空）: EXP_NAME = <训练dir>_<epoch>
#                  e.g. .../train/v2_stage1_dual/low_noise_model/epoch_1 → v2_stage1_dual_epoch_1
#   single 模式（只有 FT_MODEL_DIR）     : EXP_NAME = <训练dir>_<epoch>
#                  e.g. .../train/v2_stage1/epoch_2                      → v2_stage1_epoch_2
#   baseline（均为空）                  : EXP_NAME = baseline
if [ -n "${FT_HIGH_MODEL_DIR}" ]; then
    _train_dir="$(basename "$(dirname "$(dirname "${FT_MODEL_DIR}")")")"
    _epoch="$(basename "${FT_MODEL_DIR}")"
    EXP_NAME="${_train_dir}_${_epoch}"
elif [ -n "${FT_MODEL_DIR}" ]; then
    _train_dir="$(basename "$(dirname "${FT_MODEL_DIR}")")"
    _epoch="$(basename "${FT_MODEL_DIR}")"
    EXP_NAME="${_train_dir}_${_epoch}"
else
    EXP_NAME="baseline"
fi

# clip 名称取自 ACTION_PATH 目录名，便于 baseline vs 微调结果直接对比
CLIP_NAME="$(basename "${ACTION_PATH}")"
SAVE_FILE="${OUTPUT_BASE}/inference/${EXP_NAME}/${CLIP_NAME}.mp4"

# ---------- 日志目录 & 日志文件 ----------
LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "${SAVE_FILE}")"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 推理 v2 启动"
echo "  EXP_NAME   : ${EXP_NAME}"
echo "  CLIP_NAME  : ${CLIP_NAME}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  CKPT_DIR   : ${CKPT_DIR}"
echo "  IMAGE      : ${IMAGE}"
echo "  ACTION_PATH: ${ACTION_PATH}"
echo "  NUM_CLIPS  : ${NUM_CLIPS}"
echo "  SAVE_FILE  : ${SAVE_FILE}"
echo "  LOG_FILE   : ${LOG_FILE}"
echo "====================================================="

# ---------- 拼接推理命令 ----------
CMD=(
    python "${PROJECT_ROOT}/src/pipeline/infer_v2.py"
    --ckpt_dir           "${CKPT_DIR}"
    --image              "${IMAGE}"
    --action_path        "${ACTION_PATH}"
    --save_file          "${SAVE_FILE}"
    --prompt             "${PROMPT}"
    --frame_num          "${FRAME_NUM}"
    --num_clips          "${NUM_CLIPS}"
    --sample_steps       "${SAMPLE_STEPS}"
    --sample_shift       "${SAMPLE_SHIFT}"
    --guide_scale        "${GUIDE_SCALE}"
    --size               "${SIZE}"
    --memory_max_size    "${MEMORY_MAX_SIZE}"
    --low_model_device   "${LOW_MODEL_DEVICE}"
    --high_model_device  "${HIGH_MODEL_DEVICE}"
    --vae_device         "${VAE_DEVICE}"
    --t5_device          "${T5_DEVICE}"
)

# 可选：LoRA 权重
if [ -n "${LORA_PATH}" ]; then
    CMD+=(--lora_path "${LORA_PATH}")
fi

# 可选：全参微调 / dual 模型目录
if [ -n "${FT_MODEL_DIR}" ]; then
    CMD+=(--ft_model_dir "${FT_MODEL_DIR}")
fi
if [ -n "${FT_HIGH_MODEL_DIR}" ]; then
    CMD+=(--ft_high_model_dir "${FT_HIGH_MODEL_DIR}")
fi

# 可选：启用 Memory Bank
if [ "${USE_MEMORY}" = "true" ]; then
    CMD+=(--use_memory)
fi

# 可选：将 high_noise_model 转为 WanModelWithMemory（dual 模式下必须为 true）
if [ "${CONVERT_HIGH_TO_MEMORY}" = "true" ]; then
    CMD+=(--convert_high_to_memory)
fi

echo "执行命令："
echo "${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; exit "${PIPESTATUS[0]}"
