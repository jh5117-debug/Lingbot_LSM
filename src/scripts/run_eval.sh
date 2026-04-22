#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_eval.sh
# ============================================================

TEST_IMAGES_DIR="eval_data/images/"         # 测试图片目录（.jpg/.png）
TEST_TRAJ_DIR="eval_data/trajectories/"     # 相机轨迹目录（与图片同名，后缀不限）
MODEL_CONFIG="eval_model_configs.yaml"      # 模型配置 YAML 路径

# 评测运行名称（用于区分不同次评测，建议填入日期+简短描述）
# 留空则自动用时间戳（格式 YYYYMMDD_HHMMSS）
EVAL_RUN_NAME=""

# 跳过开关（设为 true 可单独跑推理或单独跑评分）
SKIP_INFERENCE=false   # true：跳过推理，直接对已有视频评分
SKIP_VBENCH=false      # true：跳过 VBench，只做推理

# 要评测的模型 key（空格分隔，需与 eval_model_configs.yaml 中的 key 一致）
MODELS="groupA groupB groupC"

# VBench 评测维度（对应论文 Table 2 的 6 个维度）
DIMENSIONS="imaging_quality aesthetic_quality dynamic_degree motion_smoothness temporal_flickering subject_consistency"

# 推理参数
FRAME_NUM=81
SIZE="480*832"
PROMPT="First-person view of CS:GO competitive gameplay"

CUDA_VISIBLE_DEVICES="0"   # 使用哪张 GPU，例如 "0"、"0,1"

# ============================================================
# 以下内容通常无需修改
# ============================================================

export CUDA_VISIBLE_DEVICES

# ---------- 路径计算 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EVAL_SCRIPT="${PROJECT_ROOT}/src/pipeline/eval_vbench.py"

# ---------- EVAL_RUN_NAME 默认为时间戳 ----------
if [ -z "${EVAL_RUN_NAME}" ]; then
    EVAL_RUN_NAME="$(date +%Y%m%d_%H%M%S)"
fi

# ---------- 推理结果根目录（与 run_infer_v3.sh 一致）----------
OUTPUT_BASE="${PROJECT_ROOT}/outputs"

# ---------- OUTPUT_DIR 追加 EVAL_RUN_NAME ----------
OUTPUT_DIR="${OUTPUT_BASE}/eval_vbench/${EVAL_RUN_NAME}"

# ---------- EVAL_SCRIPT 存在性检查 ----------
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "[ERROR] eval_vbench.py 不存在：${EVAL_SCRIPT}" >&2
    echo "请确认项目目录结构完整。" >&2
    exit 1
fi

# ---------- MODEL_CONFIG 路径检查 ----------
# 若为相对路径，则相对 PROJECT_ROOT 解析
if [[ "${MODEL_CONFIG}" != /* ]]; then
    MODEL_CONFIG="${PROJECT_ROOT}/${MODEL_CONFIG}"
fi

if [ ! -f "${MODEL_CONFIG}" ]; then
    echo "====================================================="
    echo "  [提示] 模型配置文件不存在：${MODEL_CONFIG}"
    echo ""
    echo "  请先运行以下命令自动生成配置模板："
    echo "    python ${EVAL_SCRIPT} --model_config ${MODEL_CONFIG}"
    echo ""
    echo "  生成模板后，填写各模型的 ckpt_dir、infer_script 等字段，"
    echo "  再重新运行本脚本。"
    echo "====================================================="
    exit 1
fi

# ---------- 日志目录 & 日志文件 ----------
LOG_DIR="${PROJECT_ROOT}/logs/run_eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

# ---------- OUTPUT_DIR 创建 ----------
mkdir -p "${OUTPUT_DIR}"

# ---------- TEST_IMAGES_DIR 路径解析 ----------
if [[ "${TEST_IMAGES_DIR}" != /* ]]; then
    TEST_IMAGES_DIR="${PROJECT_ROOT}/${TEST_IMAGES_DIR}"
fi

# ---------- TEST_TRAJ_DIR 路径解析 ----------
if [[ "${TEST_TRAJ_DIR}" != /* ]]; then
    TEST_TRAJ_DIR="${PROJECT_ROOT}/${TEST_TRAJ_DIR}"
fi

echo "====================================================="
echo "  LingBot-World Memory Enhancement — VBench 评测"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  EVAL_RUN_NAME   : ${EVAL_RUN_NAME}"
echo "  TEST_IMAGES_DIR : ${TEST_IMAGES_DIR}"
echo "  TEST_TRAJ_DIR   : ${TEST_TRAJ_DIR}"
echo "  OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "  MODEL_CONFIG    : ${MODEL_CONFIG}"
echo "  MODELS          : ${MODELS}"
echo "  SKIP_INFERENCE  : ${SKIP_INFERENCE}"
echo "  SKIP_VBENCH     : ${SKIP_VBENCH}"
echo "  LOG_FILE        : ${LOG_FILE}"
echo "====================================================="

# ---------- 拼接评测命令 ----------
# shellcheck disable=SC2206
MODELS_ARRAY=( ${MODELS} )
# shellcheck disable=SC2206
DIMENSIONS_ARRAY=( ${DIMENSIONS} )

CMD=(
    python "${EVAL_SCRIPT}"
    --test_images_dir  "${TEST_IMAGES_DIR}"
    --test_traj_dir    "${TEST_TRAJ_DIR}"
    --output_dir       "${OUTPUT_DIR}"
    --model_config     "${MODEL_CONFIG}"
    --models           "${MODELS_ARRAY[@]}"
    --dimensions       "${DIMENSIONS_ARRAY[@]}"
    --frame_num        "${FRAME_NUM}"
    --size             "${SIZE}"
    --prompt           "${PROMPT}"
)

if [ "${SKIP_INFERENCE}" = "true" ]; then
    CMD+=(--skip_inference)
fi

if [ "${SKIP_VBENCH}" = "true" ]; then
    CMD+=(--skip_vbench)
fi

echo "执行命令："
echo "${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; exit "${PIPESTATUS[0]}"
