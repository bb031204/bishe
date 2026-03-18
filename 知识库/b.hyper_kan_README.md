# HyperGKAN

基于论文 *Hypergraph Kolmogorov-Arnold Networks for Station-Level Meteorological Forecasting* 的 PyTorch 实现，用于站点级气象多步预测。

> 说明：在本课题/实验体系中，`hyper_kan` 被定位为**基线模型（baseline model）**，用于与后续改进模型进行统一对比。

## 项目亮点

- 双超图建模：同时构建邻域超图（地理距离）与语义超图（时间序列相似性）。
- KAN 替代 MLP：在超图卷积中可切换 KAN/MLP（支持消融）。
- 时空 Seq2Seq：HyperGKAN + GRU/LSTM 编码解码结构。
- 自动按要素配置：`Temperature / Cloud / Humidity / Wind` 自动覆盖路径、`top_k`、输出维度等。
- 训练工程化：AMP、梯度累积、梯度裁剪、Early Stopping、断点恢复。
- 推理评估完整：总体指标、逐步指标、3/6/12h 指标、基线对比、可视化图。

## 目录结构

```text
hyper_kan/
├── configs/config.yaml
├── train.py
├── predict.py
├── main.py
├── pause_resume/
├── src/
│   ├── data/
│   ├── graph/
│   ├── models/
│   ├── training/
│   └── utils/
├── outputs/
├── visuals/
└── data/cache/
```

## 环境安装

```bash
cd d:/bishe/WYB_bishe/hyper_kan
pip install -r requirements.txt
```

可选自检：

```bash
python check_installation.py
```

## 数据准备

代码会在训练开始时执行 `apply_element_settings`，自动把数据路径改为：

```text
../datasets/<data_dir>/{trn.pkl,val.pkl,test.pkl,position.pkl}
```

其中 `<data_dir>` 为：

- `Temperature -> temperature`
- `Cloud -> cloud_cover`
- `Humidity -> humidity`
- `Wind -> component_of_wind`

`pkl` 常见字段：`x`, `y`, `context`, `position`。

## 配置要点（configs/config.yaml）

1. `dataset_selection` 必须且只能有一个 `true`。
2. `data.context_features` 控制 8 个上下文特征是否拼接到输入。
3. `graph.use_cache=true` 会将超图缓存到 `data/cache/*.npz`。
4. `graph.visualize=true` 会输出超图统计图到 `visuals/<Element>/`。
5. `model.kan.use_kan=false` 可切到 MLP（消融）。

## 训练

```bash
python train.py --config configs/config.yaml
```

常用参数：

```bash
python train.py --config configs/config.yaml --gpu 0
python train.py --config configs/config.yaml --gpus 0,1 --multi_gpu
python train.py --resume outputs/<exp>/checkpoints/last.pt
```

## 预测

```bash
python predict.py --checkpoint outputs/<exp>/checkpoints/best_model.pt
```

或自动找最新实验：

```bash
python predict.py
```

## 一键流程

```bash
python main.py --config configs/config.yaml --gpu 0
```

可选：`--skip_train`、`--skip_predict`、`--resume`。

## 暂停与恢复训练

```bash
python pause_resume/pause.py --pause-time 60
python pause_resume/resume.py
```

## 训练与预测产物

`outputs/<timestamp>_<Element>/` 下典型文件：

- `config.yaml`：本次实验配置快照
- `preprocessor.pkl`：归一化/单位转换器
- `checkpoints/last.pt`
- `checkpoints/best_model.pt`
- `train_*.log`
- `loss_curve.png` / `loss_curve_final.png`
- `predictions.npz`
- `metrics.json`
- `baseline_comparison_summary.txt`
- `predictions_plot.png` 与分析图

## 说明与注意

- 未安装 `pykan` 时会自动回退到 `Linear/MLP`，训练可继续进行。
- `predict.py` 会优先读取 checkpoint 对应训练目录中的 `config.yaml`，避免配置错配。
- 多卡模式使用 `DataParallel`，超图张量会注册为模型 buffer，避免被错误切分。

