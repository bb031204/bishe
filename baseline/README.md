# Baseline Models for Meteorological Prediction

基于 libcity 框架的气象预测基线模型，用于与 HyperGKAN（基于KAN的可解释性气象预测模型）进行对比实验。

---

## 项目概述

本项目对 libcity 时空预测框架进行了适配改造，使其能直接加载 `.pkl` 格式的气象数据集（温度、湿度、云量、风速），并与 HyperGKAN 模型保持**完全一致**的数据预处理流程（StandardScaler标准化、开尔文→摄氏度转换、Context特征拼接等），确保公平对比。

---

## 目录结构

```
bjut_MTS_prediction/
├── run_baseline.py           # [主入口] 统一运行脚本（训练+评估）
├── baseline_config.yaml      # [主配置] 统一配置文件（数据集/模型/采样/训练参数）
├── run_model.py              # [旧入口] 原libcity运行脚本（CSV数据集用）
│
├── libcity/                  # 核心代码库
│   ├── data/dataset/
│   │   ├── pkl_dataset.py            # [新增] PKL数据集加载器（与HyperGKAN对齐）
│   │   ├── state_dataset.py          # 原始状态数据集基类
│   │   └── state_point_dataset.py    # 原始CSV数据集加载器
│   │
│   ├── model/prediction/             # 基线模型实现
│   │   ├── STGCN.py                  # 时空图卷积网络
│   │   ├── GWNET.py                  # Graph WaveNet
│   │   ├── AGCRN.py                  # 自适应图卷积循环网络
│   │   ├── ASTGCN.py                 # 注意力时空图卷积网络
│   │   ├── STGNN.py                  # 时空图神经网络
│   │   └── RNN.py                    # RNN/LSTM/GRU
│   │
│   ├── executor/
│   │   ├── meteo_executor.py         # [新增] 气象预测执行器（梯度累积+MAE/RMSE日志）
│   │   └── state_executor.py         # 原始执行器基类
│   │
│   └── evaluator/
│       └── state_evaluator.py        # 评估器
│
├── outputs/                  # 实验输出目录
│   └── {时间戳}_{模型}_{数据集}/
│       ├── train_*.log               # 训练日志
│       ├── config.json               # 本次实验完整配置
│       ├── model_cache/              # 模型权重
│       └── evaluate_cache/           # 评估结果（CSV + NPZ）
│
└── test/                     # 传统模型（HA/ARIMA/VAR）
```

---

## 支持的模型

| 模型 | 全称 | 类型 | 关键特点 |
|------|------|------|----------|
| **STGCN** | Spatio-Temporal Graph Conv Network | 图卷积 | Chebyshev多项式近似图卷积 + 时域卷积 |
| **GWNET** | Graph WaveNet | 图卷积 | 自适应邻接矩阵 + 空洞因果卷积 |
| **AGCRN** | Adaptive Graph Conv Recurrent Network | 图卷积+RNN | 节点嵌入 + 自适应图卷积 + GRU |
| **ASTGCN** | Attention-based STGCN | 图卷积+注意力 | 时空注意力机制 |
| **STGNN** | Spatio-Temporal Graph Neural Network | 图卷积 | MixHop多跳 + Inception时域模块 |
| **GRU** | Gated Recurrent Unit | RNN | 纯时序建模（无图结构） |
| **LSTM** | Long Short-Term Memory | RNN | 纯时序建模（无图结构） |

---

## 支持的气象数据集

| 数据集 | 描述 | 输出维度 | 预处理 | PKL路径 |
|--------|------|----------|--------|---------|
| **Temperature** | 温度预测 | 1 | K→°C + StandardScaler | `D:/bishe/WYB/temperature/` |
| **Cloud** | 云量预测 | 1 | StandardScaler | `D:/bishe/WYB/cloud_cover/` |
| **Humidity** | 湿度预测 | 1 | StandardScaler | `D:/bishe/WYB/humidity/` |
| **Wind** | 风速预测 (u+v) | 2 | StandardScaler | `D:/bishe/WYB/component_of_wind/` |

每个数据集目录包含：`trn.pkl`（训练集）、`val.pkl`（验证集）、`test.pkl`（测试集）、`position.pkl`（站点坐标）。

---

## 配置文件说明

### 修改模型参数的两个位置

#### 1. `baseline_config.yaml` — 全局实验配置（推荐优先修改此文件）

控制**数据集选择、模型选择、采样、训练超参数**：

```yaml
# 数据集选择（互斥，每次只启用一个）
dataset_selection:
  Temperature: true
  Cloud: false
  Humidity: false
  Wind: false

# 模型选择（可同时选多个，批量运行）
models:
  - STGCN
  # - GWNET
  # - AGCRN

# Context特征开关（8维，独立控制）
context_features:
  use_longitude: true     # 经度
  use_latitude: true      # 纬度
  use_altitude: true      # 海拔
  use_year: true          # 年份
  use_month: true         # 月份
  use_day: true           # 日期
  use_hour: true          # 小时
  use_region: false       # 区域标志

# 站点采样（空间维度）
num_stations: 768         # null=全部2048站点

# 样本采样（样本维度，按比例）
train_sample_ratio: 0.4   # 训练集使用40%样本
val_sample_ratio: 1.0
test_sample_ratio: 1.0

# 训练配置
training:
  batch_size: 16            # 单次batch大小
  accumulation_steps: 4     # 梯度累积步数（有效batch = 16 x 4 = 64）
  max_epoch: 100
  use_early_stop: true
  patience: 15

# 全局设置
global:
  seed: 42
  device: "cuda"
  gpu_id: 0
```

#### 2. `run_baseline.py` 中的 `MODEL_DEFAULTS` — 各模型的结构参数

控制**模型内部结构**（如隐藏层大小、卷积核数、dropout等）。位于 `run_baseline.py` 第 49~150 行：

```python
MODEL_DEFAULTS = {
    "STGCN": {
        "Ks": 3, "Kt": 3,                          # 空间/时域卷积核大小
        "blocks": [[1, 32, 64], [64, 32, 128]],     # ST-Conv块通道配置
        "dropout": 0,
        "learning_rate": 0.001,
        ...
    },
    "GWNET": {
        "dropout": 0.3, "blocks": 4, "layers": 2,
        "nhid": 32, "residual_channels": 32,         # 隐藏层维度
        "skip_channels": 256, "end_channels": 512,
        "learning_rate": 0.001,
        ...
    },
    "AGCRN": {
        "embed_dim": 10, "rnn_units": 64, "num_layers": 2,
        "learning_rate": 0.003,
        ...
    },
    # ... 其他模型类似
}
```

### 参数修改对照表

| 修改目的 | 修改文件 | 位置 |
|----------|----------|------|
| 切换数据集 | `baseline_config.yaml` | `dataset_selection` |
| 选择运行哪些模型 | `baseline_config.yaml` | `models` |
| 调整batch size / 梯度累积 | `baseline_config.yaml` | `training.batch_size` / `training.accumulation_steps` |
| 调整学习率 | `run_baseline.py` | `MODEL_DEFAULTS` → 对应模型的 `learning_rate` |
| 调整模型隐藏层大小 | `run_baseline.py` | `MODEL_DEFAULTS` → 对应模型的结构参数 |
| 调整epoch数 / early stop | `baseline_config.yaml` | `training.max_epoch` / `training.patience` |
| 调整Context特征 | `baseline_config.yaml` | `context_features` |
| 调整站点数量 | `baseline_config.yaml` | `num_stations` |
| 调整训练样本比例 | `baseline_config.yaml` | `train_sample_ratio` |
| 调整随机种子 | `baseline_config.yaml` | `global.seed` |
| 切换CPU/GPU | `baseline_config.yaml` | `global.device` |

---

## 使用说明

### 环境要求

- Python 3.9+
- PyTorch 2.8.0+ (需支持你的GPU架构)
- CUDA 12.8+（如使用GPU）
- scikit-learn, numpy, pandas, pyyaml, tqdm

### 安装

```bash
# 激活conda环境
conda activate baseline

# 安装PyTorch（RTX 5070需要CUDA 12.8+版本）
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install scikit-learn numpy pandas pyyaml tqdm
```

### 运行实验

#### 方式一：使用配置文件（推荐）

先编辑 `baseline_config.yaml`，然后运行：

```bash
python run_baseline.py
```

#### 方式二：命令行指定模型和数据集

```bash
# 跑单个模型 + 单个数据集
python run_baseline.py --model STGCN --dataset Temperature

# 指定GPU
python run_baseline.py --model GWNET --dataset Cloud --gpu_id 0

# 修改随机种子
python run_baseline.py --seed 123
```

#### 方式三：批量实验

在 `baseline_config.yaml` 中取消注释多个模型：

```yaml
models:
  - STGCN
  - GWNET
  - AGCRN
```

然后运行 `python run_baseline.py`，将按顺序依次训练所有模型。

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `baseline_config.yaml` |
| `--model` | 覆盖：只跑指定模型 | 配置文件中的所有模型 |
| `--dataset` | 覆盖：只跑指定数据集 | 配置文件中启用的数据集 |
| `--gpu_id` | 覆盖：GPU编号 | 配置文件中的值 |
| `--seed` | 覆盖：随机种子 | 配置文件中的值 |

### 输出说明

每次实验在 `outputs/` 下生成一个带时间戳的目录，例如：

```
outputs/20260228_180322_STGCN_Temperature/
├── train_20260228_180322_STGCN_Temperature.log   # 完整训练日志
├── config.json                                    # 本次实验的完整参数
├── model_cache/
│   └── STGCN_Temperature_best.m                   # 最佳模型权重
└── evaluate_cache/
    ├── *_metrics.csv                              # MAE/RMSE指标
    └── *_predictions.npz                          # 预测结果
```

日志包含每个epoch的训练损失、验证MAE/RMSE、学习率、耗时等信息，格式与 HyperGKAN 输出对齐。

### 显存不足 (OOM) 处理

8GB显存推荐配置（在 `baseline_config.yaml` 中调整）：

| 场景 | batch_size | accumulation_steps | 有效batch |
|------|------------|-------------------|-----------|
| 正常训练 | 16 | 4 | 64 |
| 显存紧张 | 8 | 8 | 64 |
| 极端情况 | 4 | 16 | 64 |

**原则**：保持 `batch_size × accumulation_steps` 不变，只调小 `batch_size`。

---

## 评估指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **MAE** | Mean Absolute Error | 平均绝对误差（越小越好） |
| **RMSE** | Root Mean Square Error | 均方根误差（越小越好） |

测试集评估输出**总体MAE/RMSE**和**每时间步MAE/RMSE**（共12步），便于分析不同预测时长的性能。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.8+ |
| 数据标准化 | scikit-learn StandardScaler |
| 图结构构建 | Haversine距离 + 高斯核 |
| 日志输出 | Python logging + tqdm进度条 |
| 配置管理 | YAML + 命令行覆盖 |

---

## 与HyperGKAN的数据对齐要点

| 对齐项 | 说明 |
|--------|------|
| 数据源 | 同一份PKL文件（`D:/bishe/WYB/` 下） |
| 标准化方法 | sklearn StandardScaler（同一实现） |
| 温度预处理 | Kelvin → Celsius 转换 |
| Context特征 | 相同的8维特征可选拼接 |
| 站点采样 | Train/Val/Test共享相同站点索引 |
| 随机种子 | 全局固定（默认42），确保可复现 |
