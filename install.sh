#!/bin/bash
# ========================================
# WYB_bishe 服务器环境安装脚本
# 适配: P100 16GB (Pascal 架构)
# CUDA: 11.8
# Python: 3.8+
# 环境: Anaconda (wyb_bishe)
# ========================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "  WYB_bishe 环境安装脚本"
echo "  目标显卡: P100 16GB"
echo "  CUDA 版本: 11.8"
echo "========================================"
echo ""

# ========================================
# 1. 检查 Python 版本
# ========================================
echo "[1/6] 检查 Python 版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "错误: 需要 Python 3.8 或更高版本"
    exit 1
fi
echo "✓ Python 版本符合要求"
echo ""

# ========================================
# 2. 检查 CUDA
# ========================================
echo "[2/6] 检查 CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "警告: 未找到 nvcc，CUDA 可能未安装"
    echo "请先安装 CUDA 11.8:"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_linux.run"
    echo "  sudo sh cuda_11.8.0_linux.run"
    echo ""
    read -p "是否继续安装 PyTorch? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "✓ CUDA 版本: $CUDA_VERSION"
fi
echo ""

# ========================================
# 3. 激活 Anaconda 环境
# ========================================
echo "[3/6] 激活 Anaconda 环境..."
CONDA_ENV_NAME="wyb_bishe"

# 初始化 conda（根据不同系统路径）
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null

# 检查环境是否存在
if ! conda env list | grep -wq "$CONDA_ENV_NAME"; then
    echo "错误: Anaconda 环境 '$CONDA_ENV_NAME' 不存在"
    echo "请先创建环境: conda create -n $CONDA_ENV_NAME python=3.9"
    exit 1
fi

# 激活环境
conda activate $CONDA_ENV_NAME
echo "✓ Anaconda 环境 '$CONDA_ENV_NAME' 已激活"
echo ""

# ========================================
# 4. 升级 pip
# ========================================
echo "[4/6] 升级 pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip 已升级"
echo ""

# ========================================
# 5. 安装 PyTorch (CUDA 11.8)
# ========================================
echo "[5/6] 安装 PyTorch (CUDA 11.8)..."
echo "这可能需要几分钟，请耐心等待..."

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "✓ PyTorch 安装完成"
echo ""

# 验证 PyTorch 安装
python -c "import torch; print('PyTorch 版本:', torch.__version__); \
    print('CUDA 可用:', torch.cuda.is_available()); \
    print('GPU 数量:', torch.cuda.device_count()); \
    print('GPU 名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""

# ========================================
# 6. 安装 PyTorch Geometric
# ========================================
echo "[6/6] 安装 PyTorch Geometric..."
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
echo "✓ PyTorch Geometric 安装完成"
echo ""

# ========================================
# 7. 安装其他依赖
# ========================================
echo "[7/6] 安装其他依赖包..."
pip install -r requirements.txt
echo "✓ 所有依赖安装完成"
echo ""

# ========================================
# 8. 验证安装
# ========================================
echo "========================================"
echo "  验证安装结果"
echo "========================================"
echo ""

echo "检查 PyTorch:"
python -c "import torch; print('✓ torch', torch.__version__)"
echo ""

echo "检查 CUDA:"
python -c "import torch; print('✓ CUDA 可用' if torch.cuda.is_available() else '✗ CUDA 不可用')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print('  GPU:', torch.cuda.get_device_name(0))"
fi
echo ""

echo "检查 PyTorch Geometric:"
python -c "import torch_geometric; print('✓ torch_geometric', torch_geometric.__version__)"
echo ""

echo "检查 PyKAN:"
python -c "import kan; print('✓ pykan', kan.__version__)" 2>/dev/null || echo "✗ pykan 未安装"
echo ""

echo "检查其他关键包:"
python -c "import numpy, pandas, sklearn, matplotlib, yaml; print('✓ 基础库 OK')" || echo "✗ 基础库缺失"
echo ""

echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "使用说明:"
echo "  激活环境: conda activate $CONDA_ENV_NAME"
echo "  运行 HyperGKAN: cd hyper_kan && python train.py --gpu 0"
echo "  运行基线模型: cd baseline && python run_baseline.py --gpu 1"
echo ""
