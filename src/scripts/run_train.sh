#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区：修改以下变量后运行 bash run_train.sh
# ============================================================

STAGE=1                           # 训练阶段：1 或 2

DATASET_ROOT=""                   # CSGO 数据集根目录（clip 子目录均相对此路径）
METADATA_TRAIN=""                 # metadata_train.csv 的绝对路径
METADATA_VAL=""                   # metadata_val.csv 路径（可选，留空则跳过验证）
CKPT_DIR=""                       # lingbot-world 预训练权重目录（含 low_noise_model/）
OUTPUT_DIR="outputs/stage${STAGE}" # checkpoint 保存目录
RESUME_FROM=""                    # 断点续训路径（可选，留空则从头开始）

NUM_EPOCHS=10
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4
LR_DIT=1e-5                       # Stage2 DiT 学习率（Stage1 忽略此项）
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
NUM_WORKERS=4
SAVE_EVERY=500
LOG_EVERY=10
NUM_FRAMES=81
HEIGHT=480
WIDTH=832

NUM_GPUS=8                        # 实际使用的 GPU 数量

# ============================================================
# 以下内容通常无需修改
# ============================================================

# ---------- 路径检查 ----------
_err=0
if [ -z "${DATASET_ROOT}" ]; then
    echo "[ERROR] DATASET_ROOT 未设置，请在用户配置区填写数据集根目录" >&2
    _err=1
fi
if [ -z "${METADATA_TRAIN}" ]; then
    echo "[ERROR] METADATA_TRAIN 未设置，请在用户配置区填写 metadata_train.csv 路径" >&2
    _err=1
fi
if [ -z "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR 未设置，请在用户配置区填写 lingbot-world 权重目录" >&2
    _err=1
fi
if [ "${_err}" -ne 0 ]; then
    exit 1
fi

# ---------- 根据 STAGE 选择 accelerate 配置文件 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ "${STAGE}" -eq 1 ]; then
    ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/accelerate_stage1.yaml"
elif [ "${STAGE}" -eq 2 ]; then
    ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/accelerate_stage2.yaml"
else
    echo "[ERROR] STAGE 必须为 1 或 2，当前值：${STAGE}" >&2
    exit 1
fi

if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "[ERROR] accelerate 配置文件不存在：${ACCEL_CONFIG}" >&2
    exit 1
fi

echo "====================================================="
echo "  LingBot-World Memory Enhancement 训练启动"
echo "  Stage         : ${STAGE}"
echo "  Accelerate cfg: ${ACCEL_CONFIG}"
echo "  num_processes : ${NUM_GPUS}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "====================================================="

mkdir -p "${OUTPUT_DIR}"

# ---------- 拼接 accelerate launch 命令 ----------
CMD=(
    accelerate launch
    --config_file "${ACCEL_CONFIG}"
    --num_processes "${NUM_GPUS}"
    "${PROJECT_ROOT}/src/pipeline/train.py"
    --dataset_root    "${DATASET_ROOT}"
    --metadata_train  "${METADATA_TRAIN}"
    --ckpt_dir        "${CKPT_DIR}"
    --output_dir      "${OUTPUT_DIR}"
    --stage           "${STAGE}"
    --num_epochs      "${NUM_EPOCHS}"
    --batch_size      "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --lr              "${LR}"
    --lr_dit          "${LR_DIT}"
    --weight_decay    "${WEIGHT_DECAY}"
    --max_grad_norm   "${MAX_GRAD_NORM}"
    --num_workers     "${NUM_WORKERS}"
    --save_every      "${SAVE_EVERY}"
    --log_every       "${LOG_EVERY}"
    --num_frames      "${NUM_FRAMES}"
    --height          "${HEIGHT}"
    --width           "${WIDTH}"
)

# 可选参数：METADATA_VAL（验证集）
if [ -n "${METADATA_VAL}" ]; then
    CMD+=(--metadata_val "${METADATA_VAL}")
fi

# 可选参数：RESUME_FROM（断点续训）
if [ -n "${RESUME_FROM}" ]; then
    CMD+=(--resume_from "${RESUME_FROM}")
fi

echo "执行命令："
echo "${CMD[*]}"
echo ""

exec "${CMD[@]}"
