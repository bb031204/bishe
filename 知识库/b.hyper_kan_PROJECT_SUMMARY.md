# HyperGKAN 项目总结

## 1. 项目目标

本项目实现了一个面向站点级气象预测的时空深度学习框架：**HyperGKAN**。核心目标是将超图关系建模（空间邻近 + 语义相似）与 KAN 非线性表征能力结合，在多要素、多站点场景下完成多步预测。

补充定位：在本课题/实验体系中，`hyper_kan` 作为**基线模型（baseline model）**使用，后续新模型应在相同数据与评估协议下与其对比。

## 2. 已实现能力

- 数据层：
  - 兼容多种 pkl 结构（dict / tuple / ndarray）。
  - 支持 context 特征选择性拼接（8 维可开关）。
  - 支持样本比例采样与站点采样。
- 预处理层：
  - 支持温度 K->C 转换。
  - 支持 weather/context 分离标准化与反标准化。
- 图构建层：
  - 邻域超图：基于地理位置的 KNN（支持球面距离），带缓存。
  - 语义超图：基于历史窗口相似度（euclidean/pearson/cosine），带缓存。
- 模型层：
  - `HyperGKANConv` / `DualHyperGKANConv`。
  - `HyperGKAN`（HyperGKAN Layer + GRU/LSTM 编解码）。
  - KAN/MLP 可切换，支持显存保护与降级策略。
- 训练层：
  - AMP、梯度累积、梯度裁剪、Early Stopping、学习率调度。
  - checkpoint 管理（`last.pt` / `best_model.pt`）。
  - 定时暂停、外部暂停标志、断点恢复。
- 推理与评估层：
  - 自动加载最新 checkpoint 或指定 checkpoint。
  - 输出 overall / by_step / by_horizon 指标。
  - 内置基线对比表与可视化图。

## 3. 核心技术设计

### 3.1 双超图机制

- 邻域超图关注地理邻近关系，捕捉空间局部依赖。
- 语义超图关注时间序列模式相似关系，捕捉跨区域相似行为。
- 二者在 `DualHyperGKANConv` 中融合（`add/concat/attention`）。

### 3.2 KAN 替代线性层

- 在超图卷积变换中可使用 KAN。
- 若 `pykan` 不可用或 OOM，具备自动退化机制，保证流程可运行。

### 3.3 数据集自适应配置

通过 `src/data/element_settings.py`，按 `dataset_selection` 自动覆盖：

- 数据路径
- 输出维度
- `neighbourhood_top_k` / `semantic_top_k`
- 温度转换与标准化策略
- 数值稳定参数（如 `degree_clamp_min`）

## 4. 运行链路

1. `train.py`
   - 读取配置 -> 校验并应用元素配置 -> 加载数据 -> 预处理
   - 构建/加载超图 -> 创建 DataLoader -> 创建模型/优化器/损失
   - 训练 + 验证 + checkpoint 保存
2. `predict.py`
   - 自动或手动定位 checkpoint
   - 读取训练目录配置与预处理器
   - 重建超图与模型 -> 推理 -> 反标准化 -> 指标与可视化输出
3. `main.py`
   - 串联训练与预测的一键流程

## 5. 主要输出

实验目录：`outputs/<timestamp>_<Element>/`

- `config.yaml`
- `preprocessor.pkl`
- `checkpoints/last.pt`
- `checkpoints/best_model.pt`
- `train_*.log`
- `loss_curve*.png`
- `predictions.npz`
- `metrics.json`
- `baseline_comparison_summary.txt`
- `predictions_plot*.png`

## 6. 当前状态评估

- 代码主流程完整，可完成训练-评估-可视化闭环。
- 工程可维护性较好，模块划分清晰（`data/graph/models/training/utils`）。
- 文档与代码已对齐，关键参数与命令可直接复现。

## 7. 后续建议

- 增加单元测试与集成测试（尤其是数据形状与 checkpoint 兼容性）。
- 增加更系统的多卡训练策略（DDP）与实验管理（如 TensorBoard 统一记录）。
- 将基线结果外置到配置文件，降低 `predict.py` 中硬编码比例。
