#!/usr/bin/env bash
# run_train_v3.sh — PENDING
# 等待 train_v3.py 实现完成后配套更新
# 双模型交替训练：low_noise_model（WanModelWithMemory）+ high_noise_model（WanModel）
# 等待 train_v3_stage1.py 和 train_v3_stage2.py 实现后更新
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs/run_train_v3"

echo "[ERROR] train_v3.py 尚未实现，请使用 run_train_v2.sh" >&2
exit 1
