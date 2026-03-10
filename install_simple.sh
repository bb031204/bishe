#!/bin/bash
# ========================================
# WYB_bishe 简化版安装脚本（清华源）
# ========================================

set -e

echo "========================================"
echo "  WYB_bishe 简化安装（清华源）"
echo "========================================"
echo ""

CONDA_ENV_NAME="wyb_bishe"

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null

# 检查环境是否存在
if ! conda env list | awk '{print $1}' | grep -q "^$CONDA_ENV_NAME$"; then
    echo "错误: 环境 '$CONDA_ENV_NAME' 不存在"
    echo ""
    echo "请先创建环境（清华源）:"
    echo "  conda create -n $CONDA_ENV_NAME python=3.9 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
    echo ""
    echo "或者永久配置清华源:"
    echo "  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
    echo "  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free"
    echo "  conda create -n $CONDA_ENV_NAME python=3.9"
    exit 1
fi

echo "激活环境: $CONDA_ENV_NAME"
conda activate $CONDA_ENV_NAME
echo ""

# 升级 pip（清华源）
echo "升级 pip（清华源）..."
pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

# 安装 PyTorch（必须用官方源，CUDA 版本）
echo "安装 PyTorch (CUDA 11.8, 官方源)..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
echo ""

# 安装 PyTorch Geometric
echo "安装 PyTorch Geometric（清华源）..."
pip install torch-geometric==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
echo ""

# 安装其他依赖（清华源）
echo "安装其他依赖（清华源）..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

echo "验证安装..."
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import torch_geometric; print('torch_geometric:', torch_geometric.__version__)"
echo ""
echo "安装完成！"
