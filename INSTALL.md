# 服务器环境配置指南

## 硬件信息
- **显卡**: NVIDIA P100 16GB (Pascal 架构, compute capability 6.0)
- **CUDA 版本**: 11.8
- **Python**: 3.8+

---

## 方法一：使用自动安装脚本（推荐）

### 1. 上传文件到服务器
```bash
# 在本地执行，将文件上传到服务器
scp -r WYB_bishe/ user@server:/path/to/
```

### 2. 在服务器上运行安装脚本
```bash
cd /path/to/WYB_bishe

# 给脚本执行权限
chmod +x install.sh

# 运行安装脚本
./install.sh
```

脚本会自动完成以下步骤：
1. 检查 Python 版本
2. 检查 CUDA 安装状态
3. 激活 Anaconda 环境 (wyb_bishe)
4. 安装 PyTorch (CUDA 11.8)
5. 安装 PyTorch Geometric
6. 安装所有依赖包
7. 验证安装结果

> **注意**：运行脚本前需先创建 Anaconda 环境：`conda create -n wyb_bishe python=3.9`

---

## 方法二：手动安装

### 步骤 1: 安装 CUDA 11.8（如果未安装）

```bash
# 下载 CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_linux.run

# 安装（只安装 toolkit，不安装驱动）
sudo sh cuda_11.8.0_linux.run --toolkit --silent --override

# 设置环境变量（添加到 ~/.bashrc）
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

### 步骤 2: 激活 Anaconda 环境

```bash
# 如果还没有创建环境，先创建
conda create -n wyb_bishe python=3.9

# 激活环境
conda activate wyb_bishe
```

### 步骤 3: 升级 pip

```bash
pip install --upgrade pip setuptools wheel
```

### 步骤 4: 安装 PyTorch (CUDA 11.8)

```bash
# 这是最重要的一步，确保安装的是 CUDA 11.8 版本
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 步骤 5: 验证 PyTorch 安装

```bash
python -c "import torch; \
    print('PyTorch 版本:', torch.__version__); \
    print('CUDA 可用:', torch.cuda.is_available()); \
    print('GPU 数量:', torch.cuda.device_count()); \
    print('GPU 名称:', torch.cuda.get_device_name(0))"
```

**预期输出:**
```
PyTorch 版本: 2.0.1+cu118
CUDA 可用: True
GPU 数量: 6
GPU 名称: Tesla P100-PCIE-16GB
```

### 步骤 6: 安装 PyTorch Geometric

```bash
# 安装 torch-geometric
pip install torch-geometric==2.3.1

# 安装扩展包（CUDA 11.8 版本）
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### 步骤 7: 安装其他依赖

```bash
pip install -r requirements.txt
```

---

## 依赖包列表说明

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.0.1+cu118 | 深度学习框架（P100 CUDA 11.8 版本） |
| torchvision | 0.15.2 | 视觉处理 |
| torchaudio | 2.0.2 | 音频处理 |
| pykan | 0.2.0+ | KAN 实现（HyperGKAN） |
| torch-geometric | 2.3.1 | 图神经网络库（基线模型） |
| torch-scatter | 2.1.1+ | Scatter 操作 |
| torch-sparse | 0.6.17+ | 稀疏矩阵操作 |
| dgl | 1.0.2 | Deep Graph Library |
| numpy | 1.24.4 | 数值计算 |
| pandas | 2.0.0 | 数据处理 |
| scikit-learn | 1.3.0 | 机器学习工具 |
| matplotlib | 3.7.0+ | 可视化 |
| tensorboard | 2.13.0 | 训练监控 |
| geopy | 2.3.0 | 地理距离计算 |

---

## 常见问题

### Q1: pip 安装 PyTorch 时连接超时
**解决方案**: 使用国内镜像源
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --index-url https://download.pytorch.org/whl/cu118
```

### Q2: torch-scatter/torch-sparse 安装失败
**解决方案**: 确保安装的版本与 torch 版本匹配
```bash
# 查看已安装的 torch 版本
python -c "import torch; print(torch.__version__)"

# 根据版本选择对应的 torch-scatter 版本
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### Q3: 显存不足 (OOM)
**解决方案**: 降低 batch_size 或使用梯度累积
- 修改 `hyper_kan/configs/config.yaml`: `batch_size: 16`
- 修改 `baseline/baseline_config.yaml`: `batch_size: 16`, `accumulation_steps: 4`

### Q4: CUDA 不可用
**解决方案**:
1. 检查 CUDA 是否安装: `nvcc --version`
2. 检查驱动是否安装: `nvidia-smi`
3. 确认安装的是 CUDA 版本的 PyTorch: `python -c "import torch; print(torch.version.cuda)"`

---

## 验证完整安装

运行以下命令验证所有关键包：

```bash
python -c "
import torch
import torch_geometric
import kan
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import yaml

print('✓ 所有关键包安装成功!')
print(f'  torch: {torch.__version__}')
print(f'  torch_geometric: {torch_geometric.__version__}')
print(f'  kan: {kan.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

---

## 开始运行实验

### HyperGKAN 训练
```bash
cd hyper_kan
python train.py --gpu 0
```

### 基线模型运行
```bash
cd baseline
python run_baseline.py --gpu 1
```

---

## 环境激活命令

每次使用前先激活 Anaconda 环境：
```bash
conda activate wyb_bishe
```

建议将此命令添加到 `~/.bashrc` 中自动加载：
```bash
echo 'alias wyb_env="conda activate wyb_bishe"' >> ~/.bashrc
source ~/.bashrc
```

之后只需输入 `wyb_env` 即可激活环境。
