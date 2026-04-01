#!/usr/bin/env bash
# run_infer_v3.sh — PENDING
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs/run_infer_v3"

echo "[ERROR] infer_v3.py 尚未实现，请使用 run_infer_v2.sh" >&2
exit 1
