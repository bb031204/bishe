# HyperKAN - 超图Kolmogorov-Arnold网络用于气象预测

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于论文 **"Hypergraph Kolmogorov–Arnold Networks for Station-Level Meteorological Forecasting"** 的 PyTorch 实现。

## 核心特性

- **双超图架构**: 邻域超图（地理距离）+ 语义超图（时间序列相似度）
- **KAN替代MLP**: 使用可学习样条函数提升非线性表达能力
- **Seq2Seq时序建模**: GRU Encoder-Decoder 架构
- **智能超图缓存**: 参数不变时自动复用，加速 370 倍 ⚡
- **气象要素自适应配置**: 根据不同气象要素自动应用科学配置
- **训练暂停恢复**: 支持中断后继续训练
- **定时自动暂停**: 设置训练时长，到时自动保存并暂停
- **完整评估流程**: 多指标评估 + 可视化 + 基线对比

---

## 快速开始

### 安装

```bash
# 克隆仓库
cd D:\bishe\code\hyper_kan

# 安装依赖
pip install -r requirements.txt
```

**核心依赖**:
- PyTorch >= 2.0.0
- pykan >= 0.2.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0
- matplotlib >= 3.7.0

### 准备数据

将数据放置在以下位置：

```
D:\bishe\WYB\
├── temperature\
│   ├── trn.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── position.pkl
├── cloud\
├── humidity\
└── wind\
```

**数据格式**:
- `.pkl` 文件包含 `{'x': ..., 'y': ..., 'context': ..., 'position': ...}`
- `x`: (T, N, F) - 时间步 × 站点数 × 特征维度
- `position`: (N, 2) - [latitude, longitude]

### 训练

```bash
# 基础训练
python train.py --config configs/config.yaml

# 或使用主程序
python main.py train --config configs/config.yaml

# 恢复训练
python train.py --resume outputs/xxx/best_model.pt
# 或
python pause_resume/resume.py
```

### 预测

```bash
# 自动使用最新 checkpoint
python predict.py

# 指定 checkpoint
python predict.py --checkpoint outputs/xxx/best_model.pt
```

---

## 项目结构

```
hyper_kan/
├── configs/                    # 配置文件
│   └── config.yaml            # 主配置文件
├── src/                        # 源代码
│   ├── data/                  # 数据处理模块
│   │   ├── pkl_loader.py      # PKL 数据加载
│   │   ├── dataset.py         # PyTorch 数据集
│   │   ├── preprocessing.py   # 数据预处理
│   │   └── element_settings.py # 气象要素科学配置
│   ├── graph/                 # 超图构建模块
│   │   ├── hypergraph_nei.py  # 邻域超图
│   │   ├── hypergraph_sem.py  # 语义超图
│   │   └── hypergraph_utils.py # 超图工具
│   ├── models/                # 模型架构
│   │   ├── kan_layer.py       # KAN 层实现
│   │   ├── hypergkan_conv.py  # 超图卷积层
│   │   └── hypergkan_model.py # 完整模型
│   ├── training/              # 训练模块
│   │   └── trainer.py         # 训练器
│   └── utils/                 # 工具模块
│       ├── metrics.py         # 评估指标
│       ├── logger.py          # 日志配置
│       ├── checkpoint.py      # Checkpoint 管理
│       └── visualization.py   # 可视化工具
├── pause_resume/              # 暂停恢复机制
│   ├── pause.py
│   ├── resume.py
│   └── README.md
├── test/                      # 测试脚本
├── data/                      # 数据目录
│   └── cache/                 # 超图缓存
├── visuals/                   # 可视化输出
├── outputs/                   # 训练输出
├── main.py                    # 主程序入口
├── train.py                   # 训练脚本
├── predict.py                 # 预测脚本
├── visualize_hypergraph.py    # 超图可视化
├── requirements.txt           # 依赖列表
├── README.md                  # 本文件
└── PROJECT_SUMMARY.md         # 项目总结
```

---

## 配置说明

### 数据集选择

在 `configs/config.yaml` 中选择要训练的气象要素：

```yaml
dataset_selection:
  Temperature: false
  Cloud: true        # 训练云量预测
  Humidity: false
  Wind: false
```

### 关键配置参数

```yaml
data:
  input_window: 12      # 输入时间步
  output_window: 12     # 预测时间步
  batch_size: 16
  num_stations: 768     # 站点采样数

graph:
  neighbourhood:
    top_k: 3           # 超边大小（由 element_settings 自动设置）
  semantic:
    similarity: "euclidean"  # euclidean / pearson / cosine

model:
  kan:
    use_kan: true       # true: KAN, false: MLP（消融实验）
    grid_size: 3
    spline_order: 3
  hypergkan_layer:
    num_layers: 1
    fusion_method: "add"  # concat / add / attention

training:
  time_limit_minutes: null  # 定时自动暂停（null 表示不限制）
  optimizer:
    type: "adam"
    lr: 0.01
  epochs: 100
  early_stopping:
    patience: 10
  use_amp: true        # 混合精度训练
```

---

## 气象要素科学配置

项目根据不同气象要素的特点，自动应用科学配置：

| 要素 | K 值 | 输出维度 | 特殊处理 |
|------|------|----------|----------|
| Temperature | 3 | 1 | K → °C 转换 |
| Cloud | 3 | 1 | 比例值 [0,1] |
| Humidity | 2 | 1 | 相对湿度 % |
| Wind | 3 | 2 | u/v 分量 |

**科学依据**: 论文 Section 4.4 - 不同要素对 K 的敏感度不同

---

## 高级功能

### 1. 智能超图缓存 ⚡

**性能提升**:
- 首次训练: ~8 分钟（2048 个站点）
- 第二次训练: ~1 秒（加速 370 倍）

**配置**:
```yaml
graph:
  use_cache: true
  cache_dir: "data/cache"
```

**管理**:
```bash
# 查看缓存
ls data/cache/

# 清理缓存
rm -rf data/cache/*
```

### 2. 超图可视化

**配置**:
```yaml
graph:
  visualize: true
  visual_dir: "visuals"
```

**查看**:
```
visuals/
└── Temperature/
    ├── hypergraph_neighbourhood_K3_knn.png
    └── hypergraph_semantic_K3_euclidean.png
```

### 3. 定时自动暂停

**配置**:
```yaml
training:
  time_limit_minutes: 100  # 100 分钟后自动暂停
```

### 4. 训练暂停与恢复

详见 [`pause_resume/README.md`](pause_resume/README.md)

```bash
# 快速恢复
python pause_resume/resume.py
```

---

## 消融实验

### 禁用 KAN（使用 MLP）

```yaml
model:
  kan:
    use_kan: false
```

### 禁用语义超图

```yaml
ablation:
  disable_semantic: true
```

### 禁用邻域超图

```yaml
ablation:
  disable_neighbourhood: true
```

### K 值实验

```yaml
graph:
  neighbourhood:
    top_k: 2  # 尝试 K = 2, 3, 4, 5, 6
```

---

## 评估指标

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差

**按预测步长评估**: Horizon 3 / 6 / 12

---

## 训练监控

训练过程中自动保存：

- **Checkpoint**: `outputs/<timestamp>/checkpoint_epoch_<N>.pt`
- **最佳模型**: `outputs/<timestamp>/best_model.pt`
- **损失曲线**: `outputs/<timestamp>/loss_curve.png`
- **训练日志**: `outputs/<timestamp>/train_<timestamp>.log`

---

## 常见问题

### CUDA Out of Memory

- 减小 `batch_size`
- 减小 `d_model` 或 `hidden_channels`
- 使用更少的超图卷积层

### pykan 导入失败

```bash
pip install pykan --upgrade
```

如果仍失败，模型会自动降级为标准 MLP。

### 训练不收敛

- 学习率过大 → 尝试 `lr=0.001`
- KAN 层过深 → 减少层数
- 数据未归一化 → 检查配置

---

## 模型架构

```
输入 (B, T_in, N, F)
  ↓
Input Projection (Linear)
  ↓
[HyperGKAN Encoder Layers]
  ├─ DualHyperGKANConv (Nei + Sem → Fuse)
  └─ Residual + LayerNorm
  ↓
GRU Encoder
  ↓
[Hidden State]
  ↓
GRU Decoder
  ↓
[HyperGKAN Decoder Layers]
  ↓
Output Projection (Linear)
  ↓
输出 (B, T_out, N, output_dim)
```

---

## 参考文献

```bibtex
@article{tang2024hypergkan,
  title={Hypergraph Kolmogorov–Arnold Networks for Station-Level Meteorological Forecasting},
  author={Tang, Jian and Ma, Kai},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系我
任何问题或者建议，请联系 martin_2003@163.com

---

## 致谢

- 论文作者: Jian Tang, Kai Ma
- pykan 库: [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)
- 数据集: WeatherBench
