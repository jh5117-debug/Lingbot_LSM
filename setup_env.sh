#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — LingBot-World Memory Enhancement 一键环境搭建脚本
#
# 用法:
#   bash setup_env.sh
#
# 前提条件:
#   - 已安装 conda（Miniconda / Anaconda / Miniforge 均可）
#   - 无需预先激活任何环境，脚本自动创建并使用 lingbot-lsm 环境
#
# CUDA 版本说明:
#   默认 TORCH_CUDA_TAG="cu124"，兼容 CUDA 12.x（含服务器上的 CUDA 12.8）
#   如需切换，在脚本顶部修改 TORCH_CUDA_TAG：
#     CUDA 12.4 → cu124（默认）
#     CUDA 12.1 → cu121
#     CUDA 11.8 → cu118
# =============================================================================

set -e

# =============================================================================
# Step 0 — 环境名称变量
# =============================================================================
ENV_NAME="lingbot-lsm"
PYTHON_VERSION="3.11"
TORCH_CUDA_TAG="cu124"   # 可改为 cu121 / cu118

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LINGBOT_WORLD_DIR="${SCRIPT_DIR}/refs/lingbot-world"

echo ""
echo "=== 环境配置 ==="
echo "  ENV_NAME      : ${ENV_NAME}"
echo "  PYTHON        : ${PYTHON_VERSION}"
echo "  TORCH_CUDA_TAG: ${TORCH_CUDA_TAG}"
echo "  SCRIPT_DIR    : ${SCRIPT_DIR}"
echo ""

# =============================================================================
# Step 1/7 — 创建 conda 环境
# =============================================================================
echo "=== [Step 1/7] 检查并创建 conda 环境 ${ENV_NAME} ==="

if ! command -v conda &> /dev/null; then
    echo "ERROR: 未找到 conda 命令。请先安装 Miniconda / Anaconda / Miniforge。"
    echo "       参考: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "  [INFO] 环境 '${ENV_NAME}' 已存在，跳过创建。"
else
    echo "  [INFO] 创建新环境 '${ENV_NAME}'（Python ${PYTHON_VERSION}）..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo "  [OK] 环境创建完成。"
fi

# =============================================================================
# Step 2/7 — 获取环境内 pip / python 绝对路径
# =============================================================================
echo ""
echo "=== [Step 2/7] 获取环境内 pip / python 路径 ==="

PIP="$(conda run -n "${ENV_NAME}" which pip)"
PYTHON="$(conda run -n "${ENV_NAME}" which python)"

echo "  pip    -> ${PIP}"
echo "  python -> ${PYTHON}"

# =============================================================================
# Step 3/7 — 安装 PyTorch
# =============================================================================
echo ""
echo "=== [Step 3/7] 安装 PyTorch（torch>=2.4.0, torchvision, torchaudio），CUDA=${TORCH_CUDA_TAG} ==="

"${PIP}" install \
    "torch>=2.4.0" \
    "torchvision>=0.19.0" \
    "torchaudio" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

echo "  [OK] PyTorch 安装完成。"

# =============================================================================
# Step 4/7 — 安装 flash_attn（需要 GPU 节点源码编译）
# =============================================================================
echo ""
echo "=== [Step 4/7] 安装 flash_attn（需要 GPU 节点源码编译）==="
set +e
# flash_attn 预编译 wheel 与 whl/cu124 渠道的 torch（OLD ABI）不兼容
# 必须从源码编译，需要 GPU 节点和 CUDA toolkit
"${PIP}" uninstall flash_attn -y 2>/dev/null
echo "[INFO] 尝试源码编译 flash_attn（登录节点无 GPU 时会失败，属正常现象）..."
FLASH_TMPDIR="${HOME}/.pip_tmp_flash"
mkdir -p "${FLASH_TMPDIR}"
TMPDIR="${FLASH_TMPDIR}" "${PIP}" install flash_attn --no-build-isolation
FLASH_ATTN_EXIT=$?
rm -rf "${FLASH_TMPDIR}"
set -e

if [ ${FLASH_ATTN_EXIT} -ne 0 ]; then
    echo ""
    echo "[WARN] flash_attn 安装失败（登录节点无 GPU，属正常现象）。"
    echo "  请在 GPU 计算节点上手动安装："
    echo "    conda activate ${ENV_NAME}"
    echo "    pip install flash_attn --no-build-isolation"
    echo "  注意：flash_attn 只在 CUDA 测试时需要，CPU smoke_test --dry_run 不受影响。"
    echo "[INFO] 继续安装其他依赖..."
fi

# =============================================================================
# Step 5/7 — 安装 lingbot-world 其余依赖（排除 torch/torchvision/torchaudio/flash_attn）
# =============================================================================
echo ""
echo "=== [Step 5/7] 安装 lingbot-world 其余依赖（来自 requirements.txt，已排除 torch 系列和 flash_attn）==="

"${PIP}" install \
    "opencv-python>=4.9.0.80" \
    "diffusers>=0.31.0" \
    "transformers>=4.49.0,<=4.51.3" \
    "tokenizers>=0.20.3" \
    "accelerate>=1.1.1" \
    tqdm \
    "imageio[ffmpeg]" \
    easydict \
    ftfy \
    imageio-ffmpeg \
    "numpy>=1.23.5,<2" \
    scipy

echo "  [OK] lingbot-world 其余依赖安装完成。"

# =============================================================================
# Step 6/7 — editable 安装 lingbot-world（wan 包）
# =============================================================================
echo ""
echo "=== [Step 6/7] 以 editable 方式安装 lingbot-world（pip install -e）==="
echo "  lingbot-world 路径: ${LINGBOT_WORLD_DIR}"

[ -d "${LINGBOT_WORLD_DIR}" ] || { echo "[ERROR] lingbot-world 目录不存在：${LINGBOT_WORLD_DIR}"; echo "请确认已克隆项目并包含 lingbot-world 子目录"; exit 1; }
"${PIP}" install -e "${LINGBOT_WORLD_DIR}" --no-build-isolation

echo "  [OK] lingbot-world (wan) editable 安装完成。"

# =============================================================================
# Step 7/7 — 安装 memory_module 额外依赖
# =============================================================================
echo ""
echo "=== [Step 7/7] 安装 memory_module 额外依赖：einops, wandb, pytest ==="

"${PIP}" install einops wandb pytest

echo "  [OK] memory_module 额外依赖安装完成。"

# =============================================================================
# 最终验证
# =============================================================================
echo ""
echo "================================================================"
echo "安装完成，运行最终验证..."
echo "================================================================"

"${PYTHON}" -c "
import torch
print(f'[OK] torch {torch.__version__}')
print(f'[INFO] CUDA available: {torch.cuda.is_available()} (False is normal on login nodes)')
import wan
print('[OK] wan')
import einops
print('[OK] einops')
print()
print('环境安装完成！激活方式: conda activate lingbot-lsm')
" || { echo "[WARN] 验证失败，请检查上述安装步骤"; exit 1; }
