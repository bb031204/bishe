# Baseline 对比模型说明

本目录 `baseline/` 存放的是**其他基线模型（other baseline models）**的训练与评估代码，用于和 `hyper_kan`（HyperGKAN 基线）做统一协议下的对比实验。

## 1. 目录定位

- `hyper_kan/`：HyperGKAN 基线模型代码。
- `baseline/`：STGCN、GWNET、AGCRN、ASTGCN、GRU/LSTM、STGNN 等**其他基线模型**代码。

## 2. 核心入口

- `run_baseline.py`：统一实验入口（训练 + 验证 + 测试 + 汇总文件）。
- `baseline_config.yaml`：实验配置（数据集选择、模型列表、训练参数、采样策略）。
- `libcity/data/dataset/pkl_dataset.py`：PKL 数据加载与预处理（与 HyperGKAN 对齐）。
- `libcity/executor/meteo_executor.py`：气象任务执行器（日志、评估、可视化）。

## 3. 支持模型

当前 `run_baseline.py` 默认支持并提供参数模板的模型：

- `STGCN`
- `GWNET`
- `AGCRN`
- `ASTGCN`
- `GRU`（内部使用 `RNN` 类并设置 `rnn_type=GRU`）
- `LSTM`（内部使用 `RNN` 类并设置 `rnn_type=LSTM`）
- `STGNN`

## 4. 数据与预处理（与 HyperGKAN 对齐）

数据目录由代码固定为：

```text
../datasets/
```

相对 `baseline/` 即：

```text
d:/bishe/WYB_bishe/datasets/
```

支持要素：

- `Temperature`（输出维度 1，K->C）
- `Cloud`（输出维度 1）
- `Humidity`（输出维度 1）
- `Wind`（输出维度 2）

预处理要点：

- 使用 `sklearn StandardScaler` 标准化。
- 支持 8 维 context 特征开关并可拼接到输入。
- 支持样本比例采样（train/val/test）与站点采样。
- 邻接矩阵由 `position.pkl` 基于球面距离构建。

## 5. 安装

```bash
cd d:/bishe/WYB_bishe/baseline
pip install -r requirements.txt
```

## 6. 快速运行

按 `baseline_config.yaml` 批量运行：

```bash
python run_baseline.py
```

指定单模型/单要素：

```bash
python run_baseline.py --model STGCN --dataset Temperature
```

指定 GPU：

```bash
python run_baseline.py --gpu 0
python run_baseline.py --model GWNET --dataset Cloud --gpu 1
```

说明：脚本会先设置 `CUDA_VISIBLE_DEVICES`，然后在内部按单卡逻辑运行。

## 7. 配置说明（baseline_config.yaml）

关键字段：

- `dataset_selection`：四个要素中必须只激活一个。
- `models`：要批量运行的基线模型列表。
- `context_features`：8 个上下文特征开关。
- `num_stations`：站点采样数量（`null` 表示全量）。
- `train_sample_ratio/val_sample_ratio/test_sample_ratio`：样本采样比例。
- `training.batch_size`、`training.accumulation_steps`、`training.max_epoch`。
- `output.base_dir`：实验输出目录。

## 8. 输出结果

每次实验输出到：

```text
outputs/<timestamp>_<Model>_<Element>/
```

典型文件：

- `train_*.log`
- `config.json`
- `test_summary.txt`
- `model_cache/*_best.m`
- `evaluate_cache/*_predictions.npz`
- `evaluate_cache/*_metrics.csv`
- `loss_curve.png`
- `<Model>_<Element>_predictions.png`（及分析图）

## 9. 实验目的

本目录的价值是提供一套可复现、可批量运行的**其他基线模型**实现，作为 `hyper_kan` 的对照组；请在相同数据划分、采样和评估指标（MAE/RMSE）下进行公平比较。
