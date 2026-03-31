#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_infer_v2.sh
# ============================================================
CKPT_DIR=""          # 基础模型目录
IMAGE=""             # 初始帧图像路径（.jpg）
ACTION_PATH=""       # action.npy 路径
PROMPT="First-person CS:GO gameplay"
SAVE_FILE="outputs/infer_v2/output_$(date +%Y%m%d_%H%M%S).mp4"

# 微调权重（二选一）
LORA_PATH=""         # LoRA 权重路径（lora_weights.pth），留空则不使用
FT_MODEL_DIR=""      # 全参微调目录，留空则不使用

# Memory Bank
USE_MEMORY=false     # 是否启用 memory bank（训练时未启用，推理时可选）
MEMORY_MAX_SIZE=50

# 推理参数
FRAME_NUM=81
SAMPLE_STEPS=50
SAMPLE_SHIFT=10.0
GUIDE_SCALE=5.0
SIZE="480*832"

# ============================================================
# 以下内容通常无需修改
# ============================================================

# ---------- 路径检查 ----------
_err=0
if [ -z "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR 未设置，请在用户配置区填写基础模型目录" >&2
    _err=1
fi
if [ -z "${IMAGE}" ]; then
    echo "[ERROR] IMAGE 未设置，请在用户配置区填写初始帧图像路径" >&2
    _err=1
fi
if [ -z "${ACTION_PATH}" ]; then
    echo "[ERROR] ACTION_PATH 未设置，请在用户配置区填写 action.npy 路径" >&2
    _err=1
fi
if [ "${_err}" -ne 0 ]; then
    exit 1
fi

# ---------- 路径计算 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "$(dirname "${SAVE_FILE}")"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 推理 v2 启动"
echo "  CKPT_DIR   : ${CKPT_DIR}"
echo "  IMAGE      : ${IMAGE}"
echo "  ACTION_PATH: ${ACTION_PATH}"
echo "  SAVE_FILE  : ${SAVE_FILE}"
echo "====================================================="

# ---------- 拼接推理命令 ----------
CMD=(
    python "${PROJECT_ROOT}/src/pipeline/infer_v2.py"
    --ckpt_dir    "${CKPT_DIR}"
    --image       "${IMAGE}"
    --action_path "${ACTION_PATH}"
    --save_file   "${SAVE_FILE}"
    --prompt      "${PROMPT}"
    --frame_num   "${FRAME_NUM}"
    --sample_steps "${SAMPLE_STEPS}"
    --sample_shift "${SAMPLE_SHIFT}"
    --guide_scale  "${GUIDE_SCALE}"
    --size         "${SIZE}"
    --memory_max_size "${MEMORY_MAX_SIZE}"
)

# 可选：LoRA 权重
if [ -n "${LORA_PATH}" ]; then
    CMD+=(--lora_path "${LORA_PATH}")
fi

# 可选：全参微调目录
if [ -n "${FT_MODEL_DIR}" ]; then
    CMD+=(--ft_model_dir "${FT_MODEL_DIR}")
fi

# 可选：启用 Memory Bank
if [ "${USE_MEMORY}" = "true" ]; then
    CMD+=(--use_memory)
fi

echo "执行命令："
echo "${CMD[*]}"
echo ""

exec "${CMD[@]}"
