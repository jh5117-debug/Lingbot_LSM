#!/usr/bin/env bash
# run_train_v3.sh — PENDING
# 等待 train_v3.py 实现完成后配套更新
# 双模型交替训练：low_noise_model（WanModelWithMemory）+ high_noise_model（WanModel）
# 等待 train_v3_stage1.py 和 train_v3_stage2.py 实现后更新
#
# 用户配置区（脚本实现后请在此处填写）：
#   CUDA_VISIBLE_DEVICES="0,1,2,3"  # 使用哪几张 GPU，例如 "0"、"0,1"、"0,1,2,3,4,5,6,7"
#   NUM_GPUS 将自动根据 CUDA_VISIBLE_DEVICES 计算，无需手动填写
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs/run_train_v3"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

echo "[ERROR] train_v3.py 尚未实现，请使用 run_train_v2.sh" >&2
exit 1
