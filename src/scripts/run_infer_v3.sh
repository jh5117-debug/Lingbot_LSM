#!/usr/bin/env bash
# run_infer_v3.sh — PENDING
#
# 用户配置区（脚本实现后请在此处填写）：
#   CUDA_VISIBLE_DEVICES="0"  # 使用哪张 GPU 做推理，例如 "0"、"1"、"0,1"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs/run_infer_v3"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

echo "[ERROR] infer_v3.py 尚未实现，请使用 run_infer_v2.sh" >&2
exit 1
