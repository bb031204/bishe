# WYB_BISHE 项目说明（详细版）

## 1. 项目目标与范围

本项目面向多站点气象预测任务，核心是主模型 **HyperGKAN**（双超图 + KAN）与多种 baseline 的统一对比实验。  
预测对象包含温度、云量、湿度、风（u/v 分量），统一采用多步时序预测（默认输入 12 步，预测 12 步）。

---

## 2. 项目结构（聚焦核心）

```text
WYB_bishe/
├── hyper_kan/
│   ├── train.py                       # 训练入口
│   ├── configs/config.yaml            # 主配置
│   └── src/
│       ├── data/
│       │   ├── pkl_loader.py          # PKL读取
│       │   ├── dataset.py             # Dataset/DataLoader
│       │   ├── preprocessing.py       # 标准化与温度转换
│       │   └── element_settings.py    # 要素自适应配置（K值/输出维度/路径）
│       ├── graph/
│       │   ├── hypergraph_nei.py      # 邻域超图
│       │   └── hypergraph_sem.py      # 语义超图
│       ├── models/
│       │   ├── kan_layer.py           # KANLinear/KANNetwork
│       │   ├── hypergkan_conv.py      # HyperGKANConv / DualHyperGKANConv
│       │   └── hypergkan_model.py     # Seq2Seq主模型
│       └── training/trainer.py
├── baseline/
│   ├── run_baseline.py
│   └── libcity/...                    # STGCN/GWNET/AGCRN/ASTGCN/STGNN/GRU/LSTM
└── datasets/                          # 数据根目录（当前仓库内为空，运行时按配置读取外部数据）
```

---

## 3. HyperGKAN 详解

### 3.1 双超图：采用的两种经典超图

项目采用的是**经典双超图建模**：邻域超图 + 语义超图，分别建模空间邻近关系与时序语义关系。

### 3.1.1 邻域超图（Neighbourhood Hypergraph）

实现位置：`hyper_kan/src/graph/hypergraph_nei.py`

构建逻辑：

1. 输入站点坐标 `position`，形状 `(N, 2)`，列含义 `[lat, lon]`。
2. 计算站点间球面距离（Haversine）。
3. 对每个站点做 KNN，得到该站点所在超边的成员节点。
4. 形成关联矩阵 `H_nei`，形状 `(N, E)`，其中默认 `E = N`（每个站点对应一条超边）。
5. 形成超边权重 `W_nei`，形状 `(E,)`，权重由距离指数衰减均值得到（`weight_decay` 控制）。

含义：

- `H_nei[i, e] = 1` 表示节点 `i` 属于超边 `e`；
- 超边大小近似由 `top_k` 控制（包含中心站点）。

默认参数（config）：

- `method: knn`
- `use_geodesic: true`
- `weight_decay: 0.1`
- `top_k`: 由要素配置覆盖（通常 5，湿度 2）

### 3.1.2 语义超图（Semantic Hypergraph）

实现位置：`hyper_kan/src/graph/hypergraph_sem.py`

构建逻辑：

1. 输入训练数据 `train_data['x']`（3D 或 4D）。
2. 提取最近 `input_window` 时间窗（默认 12）。
3. 将每个站点时间窗展平为向量：`(N, input_window * F)`。
4. 计算站点间相似度矩阵（默认 `euclidean`，也支持 `pearson`、`cosine`）。
5. 转成距离矩阵后做 KNN。
6. 形成 `H_sem (N, E)` 与 `W_sem (E,)`，同样默认 `E = N`。

含义：

- 强调“行为相似站点”之间的高阶连接，不受地理距离直接限制；
- 对跨区域协同变化建模更直接。

默认参数（config）：

- `similarity: euclidean`
- `input_window: 12`
- `normalize_features: true`
- `top_k`: 由要素配置覆盖（通常 5，湿度 2）

### 3.1.3 要素自适应 K 值

实现位置：`hyper_kan/src/data/element_settings.py`

- Temperature / Cloud / Wind：`K_nei = 5`, `K_sem = 5`
- Humidity：`K_nei = 2`, `K_sem = 2`

该逻辑会在训练启动时由 `apply_element_settings` 自动覆盖 `config.yaml` 的 `top_k`。

---

## 3.2 超图卷积在模型中的数学形式

实现位置：`hyper_kan/src/models/hypergkan_conv.py`

单个超图卷积层核心形式：

\[
X^{(l)} = \Phi_l\left(D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2} X^{(l-1)}\right)
\]

其中：

- `X`：节点特征，形状 `(B, N, C_in)` 或 `(N, C_in)`
- `H`：关联矩阵，`(N, E)`
- `W`：超边权重，`(E,)`
- `D_v`：节点度矩阵（由 `H`、`W`计算）
- `D_e`：超边度矩阵（由 `H`计算）
- `Φ_l`：特征变换（本项目中可选 KAN 或 MLP）

数值稳定策略：

- `float32_norm`：归一化矩阵强制 float32（兼容 AMP）
- `degree_clamp_min`：度下界裁剪，避免 0 度节点导致数值异常

双超图卷积层 `DualHyperGKANConv`：

- 分别计算 `x_nei = Conv(H_nei)` 与 `x_sem = Conv(H_sem)`；
- 再融合：`concat` / `add` / `attention`；
- 当前主配置使用 `add`（显存更友好）。

---

## 3.3 KAN 机制详解

实现位置：`hyper_kan/src/models/kan_layer.py`

### 3.3.1 KAN 在本项目中的作用

项目将 KAN 用于替代超图卷积后的线性映射层，即上式中的 `Φ_l`。  
直观上，它不是固定线性权重，而是可学习样条函数映射，增强非线性表达能力。

### 3.3.2 当前 KAN 配置

- `use_kan: true`
- `grid_size: 5`
- `spline_order: 3`
- `noise_scale: 0.1`
- `base_activation: silu`
- `chunks: 1`（对应 `kan_chunk_size`，用于分块前向）

### 3.3.3 工程实现要点

1. `KANLinear` 优先调用 pykan；若环境不满足可自动退化为 `Linear`。
2. 对大批量输入支持 chunking，降低显存峰值。
3. 发生 OOM 时有 fallback 保护，避免训练直接中断。
4. 支持消融：`use_kan=false` 即切换到 MLP/Linear 路径。

---

## 4. 数据集结构与格式（详细）

## 4.1 目录级结构

每个要素目录一般包含：

- `trn.pkl`
- `val.pkl`
- `test.pkl`
- `position.pkl`

要素映射：

- Temperature -> `datasets/temperature`
- Cloud -> `datasets/cloud_cover`
- Humidity -> `datasets/humidity`
- Wind -> `datasets/component_of_wind`

注：当前仓库内 `datasets/` 目录为空，实际训练依赖外部数据目录与配置映射。

## 4.2 PKL 内容结构

主流程期望数据为字典型：

```python
{
  "x": ndarray,
  "y": ndarray,
  "context": ndarray | None,
  "position": ndarray | None
}
```

`pkl_loader.py` 兼容多种输入（dict / tuple / ndarray），但完整训练路径建议使用 dict 标准结构。

## 4.3 关键维度定义

- `S`: 样本数
- `Tin`: 输入步长（默认 12）
- `Tout`: 输出步长（默认 12）
- `N`: 站点数（常见 2048，可采样）
- `F`: 气象特征维度
- `C`: context 维度

常见形状：

- `x`: `(S, Tin, N, F)`
- `y`: `(S, Tout, N, Fo)`
- `context`: `(S, Tin, N, C)` 或可扩展形式
- `position`: `(N, 2)`

在构建语义超图时，代码会把时窗数据展平为 `(N, Tin*F)` 进行站点相似度计算。

## 4.4 特征组成（有几个特征）

### 4.4.1 输出维度（按任务）

- Temperature: `output_dim=1`
- Cloud: `output_dim=1`
- Humidity: `output_dim=1`
- Wind: `output_dim=2`（u/v）

### 4.4.2 Context 特征

支持 8 个可选 context 特征开关：

1. longitude
2. latitude
3. altitude
4. year
5. month
6. day
7. hour
8. region

当前默认配置通常是 `region=false`，即启用 7 个。

### 4.4.3 输入总维度

模型最终输入维度：

\[
input\_dim = weather\_feature\_dim + selected\_context\_dim
\]

由数据与 mask 动态决定（代码自动推断，不需要手工硬编码）。

## 4.5 数据预处理流程

1. 加载 `trn/val/test` 与 `position`
2. （可选）温度 K -> 摄氏度
3. 标准化：
   - 天气特征单独 scaler
   - context 特征单独 scaler
4. 按 context mask 选择特征
5. 拼接 context 到输入特征
6. 样本采样（按比例）
7. 站点采样（按 `num_stations`）
8. 构建 DataLoader

## 4.6 采样机制

- 样本采样：`train_sample_ratio / val_sample_ratio / test_sample_ratio`
- 站点采样：`num_stations`
- 目标：在不改动主逻辑前提下支持快速实验与显存控制

---

## 5. Baseline（简述）

baseline 基于改造后的 libcity，统一入口 `run_baseline.py`，主要模型包括：

- STGCN
- GWNET
- AGCRN
- ASTGCN
- STGNN
- GRU
- LSTM

baseline 同样使用 PKL 数据，支持 context 特征、标准化、站点采样、样本采样，尽量与 HyperGKAN 的数据处理路径对齐，以保证对比公平。

---

## 6. 总结

本项目的方法学重点不是“发明新型超图定义”，而是把两类经典超图（邻域+语义）与 KAN 非线性映射进行有效组合，并在统一数据与训练框架下完成系统化对比。  
就论文写作而言，建议在方法章节强调：  
`双超图互补性 + KAN替代线性变换 + 要素自适应K值 + 数据流程可复现`。
