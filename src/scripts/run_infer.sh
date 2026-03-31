#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区：修改以下变量后运行 bash run_infer.sh
# ============================================================

CKPT_DIR=""                  # WanModelWithMemory checkpoint 目录（含 model_weights.bin）
IMAGE=""                     # 初始帧图像路径（.jpg）
ACTION_PATH=""               # action.npy / poses.npy / intrinsics.npy 所在目录
OUTPUT_DIR="outputs/infer"   # 推理输出目录（视频保存在此目录下）
NUM_CLIPS=1                  # 生成 clip 数量（1 = 单段生成，等价于原始 generate.py）
HEIGHT=480                   # 输出视频高度（像素）
WIDTH=832                    # 输出视频宽度（像素）

# 以下为高级参数，一般不需要修改
PROMPT=""                    # 文本提示（可为空）
FRAME_NUM=81                 # 每 clip 帧数（4n+1）
MEMORY_SIZE=8                # Memory Bank 容量 K
MEMORY_TOP_K=4               # 每次检索的记忆帧数
CLIP_STRIDE=40               # 相邻 clip 起始帧偏移（视频帧数）
SAMPLING_STEPS=40            # 扩散采样步数
GUIDE_SCALE=5.0              # Classifier-Free Guidance scale
SEED=42                      # 随机种子

# ============================================================
# 以下内容通常无需修改
# ============================================================

# ---------- 路径检查 ----------
_err=0
if [ -z "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR 未设置，请在用户配置区填写 checkpoint 目录" >&2
    _err=1
fi
if [ -z "${IMAGE}" ]; then
    echo "[ERROR] IMAGE 未设置，请在用户配置区填写初始帧图像路径" >&2
    _err=1
fi
if [ -z "${ACTION_PATH}" ]; then
    echo "[ERROR] ACTION_PATH 未设置，请在用户配置区填写 action.npy 目录路径" >&2
    _err=1
fi
if [ "${_err}" -ne 0 ]; then
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${OUTPUT_DIR}"

# 自动生成输出文件名（时间戳 + clips 数量）
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${OUTPUT_DIR}/infer_${NUM_CLIPS}clips_${TIMESTAMP}.mp4"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 推理启动"
echo "  CKPT_DIR   : ${CKPT_DIR}"
echo "  IMAGE      : ${IMAGE}"
echo "  ACTION_PATH: ${ACTION_PATH}"
echo "  NUM_CLIPS  : ${NUM_CLIPS}"
echo "  OUTPUT     : ${OUTPUT_PATH}"
echo "====================================================="

# ---------- 拼接推理命令 ----------
CMD=(
    python "${PROJECT_ROOT}/src/pipeline/infer.py"
    --ckpt_dir       "${CKPT_DIR}"
    --image          "${IMAGE}"
    --action_path    "${ACTION_PATH}"
    --output_path    "${OUTPUT_PATH}"
    --num_clips      "${NUM_CLIPS}"
    --frame_num      "${FRAME_NUM}"
    --memory_size    "${MEMORY_SIZE}"
    --memory_top_k   "${MEMORY_TOP_K}"
    --clip_stride    "${CLIP_STRIDE}"
    --sampling_steps "${SAMPLING_STEPS}"
    --guide_scale    "${GUIDE_SCALE}"
    --seed           "${SEED}"
)

# 可选：文本提示
if [ -n "${PROMPT}" ]; then
    CMD+=(--prompt "${PROMPT}")
fi

echo "执行命令："
echo "${CMD[*]}"
echo ""

exec "${CMD[@]}"
