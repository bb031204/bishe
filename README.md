# HyperGKAN：基于超图 KAN 的可解释性气象预测模型

> 本科毕业设计项目 — 复现并优化论文 *"Hypergraph Kolmogorov-Arnold Networks for Station-Level Meteorological Forecasting"* (Physica A, 2025)

本项目使用**超图神经网络 (HGNN)** 与 **Kolmogorov-Arnold Networks (KAN)** 相结合的 Seq2Seq 架构，对全球 2048 个气象站点的温度、云量、湿度、风速进行 12 小时预测，并与 7 种基线模型进行对比实验。

---

## 目录

- [项目结构](#-项目结构)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [模型架构](#-模型架构)
- [论文核心参数](#-论文核心参数)
- [配置说明](#️-配置说明)
- [多卡并行策略](#-多卡并行策略)
- [基线模型](#-基线模型)
- [输出结构](#-输出结构)
- [评估指标](#-评估指标)
- [常见问题](#-常见问题)

---

## 📁 项目结构

```
WYB_bishe/
├── hyper_kan/                    # ★ 主模型：HyperGKAN
│   ├── configs/
│   │   └── config.yaml           # 全局配置文件（论文参数已对齐）
│   ├── src/
│   │   ├── data/                 # 数据加载与预处理
│   │   │   ├── dataset.py        # SpatioTemporalDataset
│   │   │   ├── element_settings.py  # 四类气象要素的科学配置（K值/预处理/路径）
│   │   │   ├── pkl_loader.py     # PKL 数据加载器
│   │   │   └── preprocessing.py  # 标准化/反标准化
│   │   ├── graph/                # 超图构建
│   │   │   ├── hypergraph_nei.py # 邻域超图（KNN + Haversine 球面距离）
│   │   │   ├── hypergraph_sem.py # 语义超图（欧氏距离相似度）
│   │   │   └── hypergraph_utils.py
│   │   ├── models/               # 模型定义
│   │   │   ├── hypergkan_model.py   # HyperGKAN 完整 Seq2Seq 架构
│   │   │   ├── hypergkan_conv.py    # 双超图 KAN 卷积层
│   │   │   └── kan_layer.py         # KAN 层（B-spline 样条实现）
│   │   ├── training/
│   │   │   └── trainer.py        # 训练器（早停/AMP/checkpoint/DataParallel兼容）
│   │   └── utils/                # 工具（日志/指标/可视化/checkpoint）
│   ├── train.py                  # 训练脚本（支持单卡/多卡）
│   ├── predict.py                # 预测脚本
│   ├── main.py                   # 统一入口 (train/predict)
│   ├── run_all_datasets.sh       # ★ 一键并行训练4个数据集
│   └── requirements.txt
│
├── baseline/                     # 基线对比模型
│   ├── libcity/                  # 改造的 LibCity 框架
│   │   ├── model/                # STGCN, GWNET, AGCRN, ASTGCN, GRU, LSTM, STGNN
│   │   ├── executor/             # 训练/评估执行器
│   │   └── data/                 # 数据集适配器
│   ├── run_baseline.py           # 基线运行脚本
│   ├── baseline_config.yaml      # 基线配置
│   └── requirements.txt
│
├── datasets/                     # 气象数据集（4类 × 2048站点）
│   ├── temperature/              # 温度 (K→°C)
│   ├── cloud_cover/              # 云量 [0,1]
│   ├── humidity/                 # 湿度 (%)
│   └── component_of_wind/        # 风速 u,v 分量 (m/s)
│
├── install.sh                    # 自动安装脚本
├── INSTALL.md                    # 详细安装指南
└── 超图kan-main.pdf              # 原始论文
```

---

## 🔧 环境要求

### 服务器配置

| 项目 | 配置 |
|------|------|
| GPU | 6 × NVIDIA Tesla P100 16GB (PCIe) |
| CUDA | 11.8 |
| Python | 3.9 (Anaconda) |
| 系统 | Linux Ubuntu |

### 快速安装

```bash
# 1. 创建环境
conda create -n wyb_bishe python=3.9
conda activate wyb_bishe

# 2. 运行自动安装脚本
chmod +x install.sh
./install.sh

# 或手动安装核心依赖
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> 详细安装步骤见 [INSTALL.md](INSTALL.md)

---

## 🚀 快速开始

### 1. 选择数据集

编辑 `hyper_kan/configs/config.yaml`，在 `dataset_selection` 中将要训练的数据集设为 `true`（每次只能选一个）：

```yaml
dataset_selection:
  Temperature: true    # ← 选择温度
  Cloud: false
  Humidity: false
  Wind: false
```

程序会根据选择自动配置数据路径、预处理参数、超图 K 值和输出维度。

### 2. 单卡训练

```bash
cd hyper_kan

# 使用 GPU 0 训练
python train.py --gpu 0

# 指定配置文件
python train.py --config configs/config.yaml --gpu 0
```

### 3. 四数据集并行训练（推荐）

```bash
cd hyper_kan
mkdir -p logs
chmod +x run_all_datasets.sh
./run_all_datasets.sh
```

该脚本会在 4 张 GPU 上同时训练 4 个数据集（详见[多卡并行策略](#-多卡并行策略)）。

### 4. 恢复训练

```bash
python train.py --resume outputs/20260310_120000_Temperature/checkpoints/last.pt --gpu 0
```

### 5. 预测与评估

```bash
# 自动使用最新训练结果
python predict.py

# 指定 checkpoint
python predict.py --checkpoint outputs/xxx/checkpoints/best_model.pt --gpu 0
```

---

## 🏗 模型架构

HyperGKAN 采用 **Seq2Seq Encoder-Decoder** 架构：

```
输入 (B, 12, N, F)
    │
    ▼
[输入投影] Linear: F → d_model(16)
    │
    ▼ ========= Encoder =========
[HyperGKAN Layer ×1]
    ├── 邻域超图卷积 (KNN 空间超图)
    ├── 语义超图卷积 (特征相似度超图)
    ├── 融合 (Add) + 残差连接
    └── LayerNorm
    │
    ▼
[Encoder GRU] hidden_size=16, 1 layer
    │
    ▼ ========= Decoder =========
[HyperGKAN Layer ×1] (结构同 Encoder)
    │
    ▼
[Decoder GRU] hidden_size=16, 1 layer
    │
    ▼
[输出投影] Linear: 16 → output_dim
    │
    ▼
输出 (B, 12, N, output_dim)
```

**核心创新**：用 **KAN (B-spline 样条函数)** 替代超图卷积中传统的线性变换 \(\Phi_l\)：

\[
X^l = \Phi_l\left(D_v^{-1/2} \, H \, W \, D_e^{-1} \, H^T \, D_v^{-1/2} \, X^{l-1}\right)
\]

其中 \(\Phi_l\) 是 KAN 层而非 MLP，通道维度变化：`d_model(16) → hidden(32) → output(16)`。

---

## 📑 论文核心参数

以下参数已严格按论文设定配置在 `config.yaml` 中：

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 16 | 训练和验证统一 |
| Max Epochs | 500 | 最大训练轮数 |
| Early Stopping | 35 epochs | 验证集无改善则停止 |
| 损失函数 | MAE (L1) | 平均绝对误差 |
| 优化器 | Adam | lr=0.01, betas=(0.9, 0.999) |
| Weight Decay | 0.0 | 不使用 L1/L2 正则化 |
| 学习率衰减 | StepLR | 每 50 epoch 衰减为 0.5 倍 |

### KAN 层参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 激活函数 | SiLU | Sigmoid Linear Unit |
| Grid Size (G) | 5 | B-spline 网格大小 |
| Spline Order (S) | 3 | 三阶样条 |
| 通道变换 | 16 → 32 → 16 | 先扩展再降维 |

### 动态超图 K 值

| 数据集 | K 值 | 科学依据 |
|--------|------|----------|
| Temperature (温度) | **5** | 大尺度大气系统影响 |
| Cloud (云量) | **5** | 区域性云系演变 |
| Wind (风速) | **5** | 大尺度大气环流 |
| Humidity (湿度) | **2** | 强局部差异性，K 过大引入噪声 (论文 Section 4.4) |

> K 值由 `src/data/element_settings.py` 自动按数据集设置，无需手动修改。

---

## ⚙️ 配置说明

### HyperGKAN 配置 (`hyper_kan/configs/config.yaml`)

| 配置项 | 说明 | 当前值 |
|--------|------|--------|
| `dataset_selection` | 数据集选择（互斥，每次一个 true） | — |
| `data.batch_size` | 批次大小 | `16` |
| `data.num_stations` | 站点数 | `null` (全部 2048) |
| `data.input_window` | 输入时间步 | `12` (12小时) |
| `data.output_window` | 预测时间步 | `12` (12小时) |
| `model.kan.grid_size` | KAN 网格大小 | `5` |
| `model.kan.spline_order` | 样条阶数 | `3` |
| `model.kan.chunks` | KAN 分块数 (显存优化) | `1` |
| `graph.conv.hidden_channels` | 超图卷积隐藏维度 | `32` |
| `graph.conv.activation` | 激活函数 | `silu` |
| `training.epochs` | 最大训练轮数 | `500` |
| `training.early_stopping.patience` | 早停耐心 | `35` |
| `training.optimizer.lr` | 学习率 | `0.01` |
| `training.optimizer.weight_decay` | 权重衰减 | `0.0` |
| `training.scheduler.type` | 学习率策略 | `step` (每50轮×0.5) |
| `training.loss.type` | 损失函数 | `mae` |
| `training.use_amp` | 混合精度训练 | `true` |

### 消融实验开关

在 `config.yaml` 的 `ablation` 部分快速切换：

```yaml
ablation:
  disable_neighbourhood: false  # w/o-Nei：关闭邻域超图
  disable_semantic: false       # w/o-Sem：关闭语义超图
  disable_kan: false            # w/o-KAN：使用MLP替代KAN
  disable_fusion: false         # 不融合双超图
```

---

## 🖥 多卡并行策略

### 推荐方案：单卡独立训练 + 多数据集并行

由于 HyperGKAN 模型较小、batch_size 仅为 16，在 P100 (PCIe 互联) 上使用 DataParallel 多卡并行单次训练**效率反而更低**（通信开销大于计算收益）。

**最优策略**：每张 GPU 独立运行一个数据集实验。

```
GPU 0 → Temperature     GPU 1 → Cloud
GPU 2 → Humidity        GPU 3 → Wind
GPU 4-5 → 空闲（可跑基线对比模型）
```

#### 一键启动

```bash
cd hyper_kan
chmod +x run_all_datasets.sh
./run_all_datasets.sh
```

脚本会自动生成 4 份数据集专用配置，分配到 GPU 0-3 后台运行。

#### 手动启动

```bash
# 终端 1: GPU 0 - Temperature
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/config_Temperature.yaml --gpu 0

# 终端 2: GPU 1 - Cloud
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/config_Cloud.yaml --gpu 0

# 终端 3: GPU 2 - Humidity
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/config_Humidity.yaml --gpu 0

# 终端 4: GPU 3 - Wind
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/config_Wind.yaml --gpu 0
```

#### 查看训练状态

```bash
# 实时日志
tail -f hyper_kan/logs/train_Temperature.log

# GPU 使用情况
watch -n 2 nvidia-smi
```

### DataParallel 多卡（可选，不推荐）

如需强制使用多卡并行训练单个实验，需显式添加 `--multi_gpu` 标志：

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --multi_gpu
```

> 注意：代码已处理了超图矩阵 `H_nei (N, E)` 在 DataParallel 下的正确性问题——超图矩阵注册为模型 buffer 而非 forward 参数传入，避免被错误地沿 batch 维切分。

---

## 📊 基线模型

### 支持的模型

| 模型 | 类型 | 说明 |
|------|------|------|
| STGCN | 图卷积 | 时空图卷积网络 |
| GWNET | 图卷积 | 图加权网络 (自适应邻接矩阵) |
| AGCRN | 图卷积 | 自适应图卷积循环网络 |
| ASTGCN | 注意力+图 | 注意力时空图卷积网络 |
| GRU | 循环网络 | 门控循环单元 |
| LSTM | 循环网络 | 长短期记忆网络 |
| STGNN | 图神经网络 | 时空图神经网络 |

### 运行基线

```bash
cd baseline

# 运行配置中所有模型
python run_baseline.py --gpu 4

# 运行单个模型 + 单个数据集
python run_baseline.py --model STGCN --dataset Temperature --gpu 4

# 多卡分配
python run_baseline.py --model STGCN --dataset Temperature --gpu 4
python run_baseline.py --model GWNET --dataset Cloud --gpu 5
```

### 基线配置 (`baseline/baseline_config.yaml`)

| 配置项 | 说明 | 当前值 |
|--------|------|--------|
| `models` | 要运行的模型列表 | STGCN, GWNET, GRU, LSTM, STGNN |
| `training.batch_size` | 批次大小 | `32` |
| `training.accumulation_steps` | 梯度累积 | `2` |
| `training.max_epoch` | 最大轮数 | `100` |
| `training.patience` | 早停耐心 | `10` |

---

## 📤 输出结构

### HyperGKAN 输出

```
hyper_kan/outputs/
└── 20260310_120000_Temperature/     # 时间戳_数据集名
    ├── config.yaml                   # 本次训练的配置备份
    ├── checkpoints/
    │   ├── best_model.pt             # 验证集最优模型
    │   ├── last.pt                   # 最后一次保存的模型 (用于恢复训练)
    │   └── checkpoint_epoch_100.pt   # 每10轮保存一次
    ├── preprocessor.pkl              # 预处理器 (预测时反标准化)
    ├── loss_curve.png                # 训练/验证损失曲线
    ├── predictions.npz               # 预测结果
    ├── metrics.json                  # 评估指标 (MAE, RMSE)
    └── train.log                     # 完整训练日志
```

### 基线输出

```
baseline/outputs/
└── 20260310_120000_STGCN_Temperature/
    ├── model_cache/
    │   └── STGCN_Temperature_best.m
    ├── evaluate_cache/
    │   ├── results.csv
    │   └── results.json
    └── test_summary.txt
```

---

## 📈 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **MAE** | \(\frac{1}{n}\sum\|y - \hat{y}\|\) | 平均绝对误差 (**主指标**) |
| **RMSE** | \(\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}\) | 均方根误差 |

> 注意：MAPE 仅作参考指标。云量和湿度数据中存在接近 0 的值，可能导致 MAPE 异常偏大。

### 实验记录模板

| 实验 | 模型 | 数据集 | GPU | Epochs | MAE | RMSE | 训练时间 |
|------|------|--------|-----|--------|-----|------|----------|
| 1 | HyperGKAN | Temperature | 0 | — | — | — | — |
| 2 | HyperGKAN | Cloud | 1 | — | — | — | — |
| 3 | HyperGKAN | Humidity | 2 | — | — | — | — |
| 4 | HyperGKAN | Wind | 3 | — | — | — | — |
| 5 | STGCN | Temperature | 4 | — | — | — | — |
| 6 | GWNET | Temperature | 5 | — | — | — | — |

---

## 🔬 科学配置说明

### 温度 (Temperature)
- **单位转换**：开尔文 → 摄氏度 (K − 273.15)
- **标准化**：Standard Scaler
- **超图 K 值**：neighbourhood = 5, semantic = 5
- **输出维度**：1

### 云量 (Cloud)
- **数据范围**：[0, 1]（分数形式）
- **标准化**：Standard Scaler
- **超图 K 值**：neighbourhood = 5, semantic = 5
- **输出维度**：1

### 湿度 (Humidity)
- **数据范围**：百分比 (%)
- **标准化**：Standard Scaler
- **超图 K 值**：neighbourhood = **2**, semantic = **2** ⭐
- **输出维度**：1
- **科学依据**：湿度具有强局部差异性，超边节点数过大会引入噪声 (论文 Section 4.4, Figure 3)

### 风速 (Wind)
- **数据形式**：u, v 两个分量 (m/s)
- **标准化**：Standard Scaler
- **超图 K 值**：neighbourhood = 5, semantic = 5
- **输出维度**：2

---

## 🐛 常见问题

### Q1: 显存不足 (OOM)

```
RuntimeError: CUDA out of memory
```

**解决方案**（按优先级）：
1. 增加 `model.kan.chunks`：`1` → `4` 或 `12`（分块处理 KAN 层，不影响精度）
2. 减少 `data.num_stations`：如 `1024` 或 `512`
3. 关闭 AMP：`training.use_amp: false`（P100 对 FP16 支持有限）
4. 降低 `data.batch_size`

### Q2: 训练速度慢

**可能原因与方案**：
- `num_workers` 过低 → 设为 `4`（已在 config 中配置）
- 数据集未缓存 → 首次运行后超图会自动缓存到 `data/cache/`
- KAN chunks 过多 → 适当减小 `chunks` 值

### Q3: GPU 指定不生效

`--gpu` 参数通过 `os.environ["CUDA_VISIBLE_DEVICES"]` 在 `import torch` **之前**设置，确保：
- 不要在脚本中提前 `import torch`
- 使用 `run_all_datasets.sh` 时通过 `CUDA_VISIBLE_DEVICES` 环境变量隔离

### Q4: 恢复训练报错

```bash
# 确保使用 last.pt（包含优化器状态），不要用 best_model.pt
python train.py --resume outputs/xxx/checkpoints/last.pt --gpu 0
```

### Q5: 数据加载失败

- 确认数据集路径：`datasets/` 下需包含 `trn.pkl`, `val.pkl`, `test.pkl`, `position.pkl`
- 检查 `element_settings.py` 中的 `DATA_ROOT` 是否指向正确的数据集目录

### Q6: NaN Loss

- 检查数据是否包含异常值
- 降低学习率：`training.optimizer.lr: 0.001`
- 增大梯度裁剪：`training.grad_clip: 1.0`
- 确保 `float32_norm: true`（在 AMP 模式下强制使用 FP32 计算超图归一化矩阵）

---

## 📚 参考文献

```
@article{HyperGKAN2025,
  title={Hypergraph Kolmogorov-Arnold Networks for Station-Level Meteorological Forecasting},
  journal={Physica A: Statistical Mechanics and its Applications},
  year={2025}
}
```

---

## 👨‍💻 作者

本科毕业设计项目 — 基于超图 KAN 的可解释性气象预测模型复现与优化

---

## 📄 许可证

本项目仅供学术研究使用。
