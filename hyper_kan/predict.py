"""
HyperGKAN 预测脚本
"""
import os
# 设置 GPU 环境变量（必须在 import torch 之前）
import sys
if '--gpu' in sys.argv:
    gpu_idx = sys.argv[sys.argv.index('--gpu') + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

import yaml
import torch
import numpy as np
import argparse
from tqdm import tqdm
from typing import Optional
from datetime import datetime

from src.data import load_pkl_data, load_position_data, create_data_loaders, DataPreprocessor
from src.data import apply_element_settings, validate_dataset_selection
from src.graph import build_neighbourhood_hypergraph, build_semantic_hypergraph
from src.models import HyperGKAN
from src.utils import compute_metrics, setup_logger, load_checkpoint, get_latest_checkpoint
from src.utils.visualization import plot_predictions, plot_metrics_comparison

# 获取脚本所在目录（用于解析相对路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ANSI颜色代码
YELLOW = '\033[93m'
GREEN = '\033[92m'
RESET = '\033[0m'


# ==================== 基线模型指标 ====================
# 数据来源:
#   - Temperature 论文基线 (MLP/LSTM/GCN/GAT/STGCN): 论文 Table 2
#   - 实验基线: D:\bishe\WYB\bjut_MTS_prediction 中的实验结果
#     - HA/ARIMA: 根目录 CSV 文件
#     - STGCN/AGCRN/GWNET/ASTGCN/RNN/GRU/LSTM: libcity evaluate_cache + log 文件
# 注: ARIMA 的 Temperature 和 Wind RMSE 列疑似为 MSE, 已取 sqrt 修正

BASELINE_METRICS = {
    'Temperature': {
        '3h': {
            # 论文基线
            'MLP': {'mae': 0.85, 'rmse': 1.12},
            'LSTM': {'mae': 0.78, 'rmse': 1.05},
            'GCN': {'mae': 0.82, 'rmse': 1.10},
            'GAT': {'mae': 0.76, 'rmse': 1.02},
            'STGCN': {'mae': 0.72, 'rmse': 0.98},
            # 实验基线
            'HA': {'mae': 8.688, 'rmse': 12.196},
            'ARIMA': {'mae': 6.204, 'rmse': 8.349},
            'RNN': {'mae': 5.850, 'rmse': 7.858},
        },
        '6h': {
            'MLP': {'mae': 1.15, 'rmse': 1.45},
            'LSTM': {'mae': 1.05, 'rmse': 1.38},
            'GCN': {'mae': 1.10, 'rmse': 1.42},
            'GAT': {'mae': 1.02, 'rmse': 1.35},
            'STGCN': {'mae': 0.98, 'rmse': 1.30},
            'HA': {'mae': 8.688, 'rmse': 12.196},
            'ARIMA': {'mae': 8.206, 'rmse': 8.375},
            'RNN': {'mae': 5.838, 'rmse': 7.843},
        },
        '12h': {
            'MLP': {'mae': 1.65, 'rmse': 2.10},
            'LSTM': {'mae': 1.52, 'rmse': 2.00},
            'GCN': {'mae': 1.58, 'rmse': 2.05},
            'GAT': {'mae': 1.48, 'rmse': 1.95},
            'STGCN': {'mae': 1.42, 'rmse': 1.88},
            'HA': {'mae': 8.688, 'rmse': 12.196},
            'ARIMA': {'mae': 11.784, 'rmse': 8.462},
            'RNN': {'mae': 5.824, 'rmse': 7.825},
        }
    },
    'Cloud': {
        '3h': {
            'HA': {'mae': 0.3223, 'rmse': 0.4463},
            'ARIMA': {'mae': 0.3522, 'rmse': 0.6554},
            'STGCN': {'mae': 0.2398, 'rmse': 0.2921},
            'AGCRN': {'mae': 0.2065, 'rmse': 0.3082},
            'GWNET': {'mae': 0.2079, 'rmse': 0.3117},
            'ASTGCN': {'mae': 0.2396, 'rmse': 0.2928},
            'RNN': {'mae': 0.2207, 'rmse': 0.3227},
        },
        '6h': {
            'HA': {'mae': 0.3223, 'rmse': 0.4463},
            'ARIMA': {'mae': 0.3829, 'rmse': 0.9436},
            'STGCN': {'mae': 0.2419, 'rmse': 0.2953},
            'AGCRN': {'mae': 0.2085, 'rmse': 0.3108},
            'GWNET': {'mae': 0.2099, 'rmse': 0.3143},
            'ASTGCN': {'mae': 0.2430, 'rmse': 0.2954},
            'RNN': {'mae': 0.2205, 'rmse': 0.3225},
        },
        '12h': {
            'HA': {'mae': 0.3223, 'rmse': 0.4463},
            'ARIMA': {'mae': 0.3402, 'rmse': 1.0085},
            'STGCN': {'mae': 0.2430, 'rmse': 0.2982},
            'AGCRN': {'mae': 0.2099, 'rmse': 0.3125},
            'GWNET': {'mae': 0.2113, 'rmse': 0.3163},
            'ASTGCN': {'mae': 0.2465, 'rmse': 0.2977},
            'RNN': {'mae': 0.2206, 'rmse': 0.3225},
        },
    },
    'Humidity': {
        '3h': {
            'HA': {'mae': 14.208, 'rmse': 19.286},
            'ARIMA': {'mae': 12.724, 'rmse': 24.274},
            'RNN': {'mae': 9.343, 'rmse': 12.861},
            'GRU': {'mae': 8.488, 'rmse': 11.699},
            'LSTM': {'mae': 8.742, 'rmse': 12.031},
            'STGCN': {'mae': 7.879, 'rmse': 10.870},
            'AGCRN': {'mae': 7.593, 'rmse': 10.547},
            'GWNET': {'mae': 7.629, 'rmse': 10.588},
            'ASTGCN': {'mae': 7.872, 'rmse': 10.621},
        },
        '6h': {
            'HA': {'mae': 14.208, 'rmse': 19.286},
            'ARIMA': {'mae': 18.143, 'rmse': 41.001},
            'RNN': {'mae': 9.333, 'rmse': 12.831},
            'GRU': {'mae': 8.478, 'rmse': 11.668},
            'LSTM': {'mae': 8.733, 'rmse': 12.004},
            'STGCN': {'mae': 8.189, 'rmse': 11.200},
            'AGCRN': {'mae': 7.798, 'rmse': 10.795},
            'GWNET': {'mae': 7.840, 'rmse': 10.837},
            'ASTGCN': {'mae': 8.105, 'rmse': 10.883},
        },
        '12h': {
            'HA': {'mae': 14.208, 'rmse': 19.286},
            'ARIMA': {'mae': 27.544, 'rmse': 81.546},
            'RNN': {'mae': 9.304, 'rmse': 12.779},
            'GRU': {'mae': 8.464, 'rmse': 11.629},
            'LSTM': {'mae': 8.710, 'rmse': 11.956},
            'STGCN': {'mae': 8.415, 'rmse': 11.419},
            'AGCRN': {'mae': 7.939, 'rmse': 10.970},
            'GWNET': {'mae': 7.985, 'rmse': 11.005},
            'ASTGCN': {'mae': 8.275, 'rmse': 11.071},
        },
    },
    'Wind': {
        '3h': {
            'HA': {'mae': 4.540, 'rmse': 6.196},
            'ARIMA': {'mae': 12.100, 'rmse': 14.565},
            'RNN': {'mae': 3.230, 'rmse': 4.392},
        },
        '6h': {
            'HA': {'mae': 4.540, 'rmse': 6.196},
            'ARIMA': {'mae': 13.218, 'rmse': 14.567},
            'RNN': {'mae': 3.229, 'rmse': 4.392},
        },
        '12h': {
            'HA': {'mae': 4.540, 'rmse': 6.196},
            'ARIMA': {'mae': 15.527, 'rmse': 14.573},
            'RNN': {'mae': 3.228, 'rmse': 4.394},
        },
    }
}


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_checkpoint() -> Optional[str]:
    """
    自动查找最新的训练checkpoint

    Returns:
        checkpoint路径，如果找不到返回None
    """
    # 使用脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, 'outputs')

    if not os.path.exists(outputs_dir):
        return None

    # 查找所有时间戳目录
    timestamp_dirs = []
    for dir_name in os.listdir(outputs_dir):
        if dir_name.count('_') >= 2:  # 格式: YYYYMMDD_HHMMSS_Element
            try:
                dir_path = os.path.join(outputs_dir, dir_name)
                if os.path.isdir(dir_path):
                    timestamp_dirs.append((dir_path, dir_name))
            except:
                pass

    if not timestamp_dirs:
        return None

    # 按时间戳排序，取最新的
    timestamp_dirs.sort(key=lambda x: x[1], reverse=True)
    latest_dir = timestamp_dirs[0][0]

    # 在该目录中查找checkpoint
    checkpoint_dir = os.path.join(latest_dir, 'checkpoints')

    if os.path.exists(checkpoint_dir):
        checkpoint = get_latest_checkpoint(checkpoint_dir)
        if checkpoint:
            return checkpoint, latest_dir

    return None, None


def compare_with_baselines(overall_metrics: dict, horizon_metrics: dict,
                           element: str, logger) -> dict:
    """
    与基线模型对比（仅计算，不输出日志）

    Args:
        overall_metrics: 整体指标
        horizon_metrics: 分时段指标
        element: 气象元素名称
        logger: 日志记录器

    Returns:
        对比结果字典，包含所有基线模型的完整信息
    """
    if element not in BASELINE_METRICS:
        logger.warning(f"警告: 元素 {element} 没有基线数据")
        return None

    comparison = {
        'element': element,
        'overall': {},
        'by_horizon': {}
    }

    hypergkan_mae = overall_metrics['mae']
    hypergkan_rmse = overall_metrics['rmse']

    # 整体对比 (12h)
    baseline_12h = BASELINE_METRICS[element]['12h']
    for model_name, metrics in baseline_12h.items():
        mae_diff = ((hypergkan_mae - metrics['mae']) / metrics['mae']) * 100
        rmse_diff = ((hypergkan_rmse - metrics['rmse']) / metrics['rmse']) * 100

        comparison['overall'][model_name] = {
            'baseline_mae': metrics['mae'],
            'baseline_rmse': metrics['rmse'],
            'hypergkan_mae': hypergkan_mae,
            'hypergkan_rmse': hypergkan_rmse,
            'mae_improvement_pct': mae_diff,
            'rmse_improvement_pct': rmse_diff
        }

    # 分时段对比
    for horizon in [3, 6, 12]:
        if horizon in horizon_metrics:
            horizon_str = f"{horizon}h"
            if horizon_str in BASELINE_METRICS[element]:
                h_metrics = horizon_metrics[horizon]
                h_baseline = BASELINE_METRICS[element][horizon_str]

                comparison['by_horizon'][horizon] = {
                    'hypergkan_mae': h_metrics['mae'],
                    'hypergkan_rmse': h_metrics['rmse'],
                    'baselines': {}
                }

                for model_name, metrics in h_baseline.items():
                    comparison['by_horizon'][horizon]['baselines'][model_name] = {
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse']
                    }

    # 计算相对于最强基线(MAE最低的模型)的改进
    best_model_name = min(baseline_12h, key=lambda m: baseline_12h[m]['mae'])
    best_baseline = baseline_12h[best_model_name]
    mae_improvement = ((hypergkan_mae - best_baseline['mae']) / best_baseline['mae']) * 100
    rmse_improvement = ((hypergkan_rmse - best_baseline['rmse']) / best_baseline['rmse']) * 100
    comparison['vs_best_baseline'] = {
        'model_name': best_model_name,
        'baseline_mae': best_baseline['mae'],
        'baseline_rmse': best_baseline['rmse'],
        'mae_improvement_pct': mae_improvement,
        'rmse_improvement_pct': rmse_improvement,
    }
    # 保留 vs_stgcn 兼容性
    if 'STGCN' in baseline_12h:
        stgcn = baseline_12h['STGCN']
        comparison['vs_stgcn'] = {
            'stgcn_mae': stgcn['mae'],
            'stgcn_rmse': stgcn['rmse'],
            'mae_improvement_pct': ((hypergkan_mae - stgcn['mae']) / stgcn['mae']) * 100
        }

    return comparison


def predict(model, test_loader, H_nei, H_sem, W_nei, W_sem, device, logger):
    """预测"""
    model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []

    logger.info("Generating predictions...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", colour='green'):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            pred = model(x, H_nei, H_sem, W_nei, W_sem, output_length=y.shape[1])

            all_inputs.append(x.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 合并结果
    inputs = np.concatenate(all_inputs, axis=0)
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    logger.info(f"Predictions generated: {predictions.shape}")

    return inputs, predictions, targets


def evaluate_predictions(predictions, targets, config, logger):
    """评估预测结果"""
    logger.info("=" * 50)
    logger.info("Evaluating predictions...")

    # 转换为torch张量
    pred_tensor = torch.from_numpy(predictions).float()
    target_tensor = torch.from_numpy(targets).float()

    # 计算整体指标
    overall_metrics = compute_metrics(
        pred_tensor,
        target_tensor,
        metrics=config['evaluation']['metrics']
    )

    logger.info("Overall Metrics:")
    for metric, value in overall_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")

    # 按单步评估 (每个预测步的 MAE/RMSE)
    T = predictions.shape[1]
    step_metrics = {}

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Test MAE:            {overall_metrics['mae']:.4f}")
    logger.info(f"  Test RMSE:           {overall_metrics['rmse']:.4f}")
    logger.info("")
    logger.info("-" * 60)
    logger.info(f"  {'Step':>6}     {'MAE':>10}    {'RMSE':>10}")
    logger.info("-" * 60)

    for t in range(T):
        pred_t = pred_tensor[:, t:t+1, :, :]
        target_t = target_tensor[:, t:t+1, :, :]
        metrics_t = compute_metrics(pred_t, target_t, metrics=['mae', 'rmse'])
        step_metrics[t + 1] = metrics_t
        logger.info(f"  {t+1:>6}    {metrics_t['mae']:>10.4f}    {metrics_t['rmse']:>10.4f}")

    logger.info("-" * 60)
    logger.info(f"  {'Avg':>6}    {overall_metrics['mae']:>10.4f}    {overall_metrics['rmse']:>10.4f}")
    logger.info("")

    # 按时间段评估 (累积 horizon)
    horizon_metrics = {}

    horizons = [3, 6, 12] if T >= 12 else [T // 3, T // 2, T]

    for h in horizons:
        if h <= T:
            pred_h = pred_tensor[:, :h, :, :]
            target_h = target_tensor[:, :h, :, :]

            metrics_h = compute_metrics(pred_h, target_h, metrics=['mae', 'rmse'])
            horizon_metrics[h] = metrics_h

            logger.info(f"Horizon {h} steps:")
            for metric, value in metrics_h.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")

    return overall_metrics, horizon_metrics, step_metrics


def save_results(inputs, predictions, targets, overall_metrics, horizon_metrics,
                output_dir, config, logger, comparison=None, step_metrics=None):
    """保存结果"""
    logger.info("=" * 50)
    logger.info("Saving results...")

    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果
    if config['output']['save_predictions']:
        np.savez(
            os.path.join(output_dir, 'predictions.npz'),
            inputs=inputs,
            predictions=predictions,
            targets=targets
        )
        logger.info(f"Predictions saved to {output_dir}/predictions.npz")

    # 保存指标
    import json

    metrics_dict = {
        'overall': overall_metrics,
        'by_horizon': {str(k): v for k, v in horizon_metrics.items()},
        'by_step': {str(k): v for k, v in (step_metrics or {}).items()},
        'baseline_comparison': comparison
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Metrics saved to {output_dir}/metrics.json")

    # 保存对比summary文本（完整版）
    if comparison is not None and 'overall' in comparison:
        summary_path = os.path.join(output_dir, 'baseline_comparison_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 75 + "\n")
            f.write("HyperGKAN vs Baseline Models - Complete Comparison Summary\n")
            f.write("=" * 75 + "\n\n")

            element = comparison.get('element', config['meta']['element'])
            hypergkan_mae = overall_metrics['mae']
            hypergkan_rmse = overall_metrics['rmse']

            f.write(f"Element: {element}\n")
            f.write(f"Prediction Horizon: 12 hours\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # TEST RESULTS - 逐步测试结果
            f.write("=" * 60 + "\n")
            f.write("TEST RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Test MAE:            {hypergkan_mae:.4f}\n")
            f.write(f"  Test RMSE:           {hypergkan_rmse:.4f}\n\n")

            if step_metrics:
                f.write("-" * 60 + "\n")
                f.write(f"  {'Step':>6}     {'MAE':>10}    {'RMSE':>10}\n")
                f.write("-" * 60 + "\n")
                for step in sorted(step_metrics.keys()):
                    sm = step_metrics[step]
                    f.write(f"  {step:>6}    {sm['mae']:>10.4f}    {sm['rmse']:>10.4f}\n")
                f.write("-" * 60 + "\n")
                f.write(f"  {'Avg':>6}    {hypergkan_mae:>10.4f}    {hypergkan_rmse:>10.4f}\n")
            f.write("\n")

            # 整体对比 (12h)
            f.write("=" * 75 + "\n")
            f.write("Overall Performance (12 hours)\n")
            f.write("=" * 75 + "\n\n")
            f.write(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'vs HyperGKAN':>18}\n")
            f.write("-" * 75 + "\n")

            for model_name, metrics in comparison['overall'].items():
                mae_diff = metrics['mae_improvement_pct']
                imp_str = f"{mae_diff:+.1f}%"
                f.write(f"{model_name:<20} {metrics['baseline_mae']:>12.4f} "
                       f"{metrics['baseline_rmse']:>12.4f} {imp_str:>18}\n")

            f.write("-" * 75 + "\n")
            f.write(f"{'HyperGKAN (Ours)':<20} {hypergkan_mae:>12.4f} "
                   f"{hypergkan_rmse:>12.4f} {'--':>18}\n\n")

            # 分时段对比
            f.write("=" * 75 + "\n")
            f.write("Performance by Prediction Horizon\n")
            f.write("=" * 75 + "\n")

            for horizon in [3, 6, 12]:
                if horizon in comparison.get('by_horizon', {}):
                    h_data = comparison['by_horizon'][horizon]
                    f.write(f"\n{horizon}h Prediction:\n")
                    f.write("-" * 75 + "\n")
                    f.write(f"{'Model':<20} {'MAE':>12} {'RMSE':>12}\n")
                    f.write("-" * 75 + "\n")

                    for model_name, metrics in h_data['baselines'].items():
                        f.write(f"{model_name:<20} {metrics['mae']:>12.4f} {metrics['rmse']:>12.4f}\n")

                    f.write(f"{'HyperGKAN':<20} {h_data['hypergkan_mae']:>12.4f} {h_data['hypergkan_rmse']:>12.4f}\n\n")

            # 总结
            f.write("=" * 75 + "\n")
            f.write("Summary\n")
            f.write("=" * 75 + "\n\n")

            if 'vs_best_baseline' in comparison:
                best_data = comparison['vs_best_baseline']
                best_name = best_data['model_name']
                mae_imp = best_data['mae_improvement_pct']
                f.write(f"HyperGKAN vs {best_name} (best baseline, lowest 12h MAE):\n")
                f.write(f"  {best_name} MAE:     {best_data['baseline_mae']:.4f}\n")
                f.write(f"  {best_name} RMSE:    {best_data['baseline_rmse']:.4f}\n")
                f.write(f"  HyperGKAN MAE: {hypergkan_mae:.4f}\n")
                f.write(f"  HyperGKAN RMSE:{hypergkan_rmse:.4f}\n")
                if mae_imp < 0:
                    f.write(f"  MAE Improvement:  {-mae_imp:.1f}% (lower is better)\n")
                    f.write(f"  Result: HyperGKAN outperforms {best_name} by {-mae_imp:.1f}%\n")
                else:
                    f.write(f"  MAE Difference:   +{mae_imp:.1f}%\n")
                    f.write(f"  Result: {best_name} performs better by {mae_imp:.1f}%\n")
                f.write("\n")

            # 排名统计
            f.write("Ranking (12h MAE, ascending):\n")
            all_models = {m: v['baseline_mae'] for m, v in comparison['overall'].items()}
            all_models['HyperGKAN'] = hypergkan_mae
            for rank, (name, mae) in enumerate(sorted(all_models.items(), key=lambda x: x[1]), 1):
                marker = " <-- Ours" if name == 'HyperGKAN' else ""
                f.write(f"  {rank}. {name:<20} MAE={mae:.4f}{marker}\n")
            f.write("\n")

        logger.info(f"Baseline comparison summary saved to {summary_path}")

        # 输出到控制台的简要摘要
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"{GREEN}Baseline Comparison Summary{RESET}")
        logger.info("=" * 60)
        logger.info(f"Overall (12h) - MAE: {hypergkan_mae:.4f}, RMSE: {hypergkan_rmse:.4f}")

        if 'vs_best_baseline' in comparison:
            best_data = comparison['vs_best_baseline']
            best_name = best_data['model_name']
            mae_imp = best_data['mae_improvement_pct']
            if mae_imp < 0:
                logger.info(f"{GREEN}Outperforms {best_name} (best baseline) by {-mae_imp:.1f}%{RESET}")
            else:
                logger.info(f"{YELLOW}Behind {best_name} (best baseline) by {mae_imp:.1f}%{RESET}")

        # 显示所有基线对比的简表
        n_better = sum(1 for m, v in comparison['overall'].items()
                       if hypergkan_mae < v['baseline_mae'])
        n_total = len(comparison['overall'])
        logger.info(f"HyperGKAN beats {n_better}/{n_total} baseline models (12h MAE)")
        logger.info(f"Full comparison saved to: {summary_path}")
        logger.info("")

    # 可视化
    if config['output']['save_plots']:
        logger.info("Generating plots...")

        # 预测曲线（智能选择样本+站点，附加综合分析图）
        plot_predictions(
            pred=predictions,
            target=targets,
            save_path=os.path.join(output_dir, 'predictions_plot.png'),
            num_samples=min(4, predictions.shape[0]),
            num_stations=min(4, predictions.shape[2]),
            horizon_steps=[3, 6, 12],
            element=config['meta']['element']
        )

        logger.info("Plots saved")


def main(args):
    """主函数"""
    # 处理checkpoint路径，优先从训练目录加载config
    checkpoint_path = args.checkpoint
    output_dir = args.output
    training_dir = None

    # 如果命令行指定了 GPU，覆盖 config 中的设置
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 推断训练目录
    if checkpoint_path is not None:
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if 'checkpoints' in checkpoint_dir:
            training_dir = os.path.dirname(checkpoint_dir)
    
    # 优先使用训练目录中保存的config（确保与checkpoint匹配）
    config_path = args.config
    if training_dir is not None:
        saved_config = os.path.join(training_dir, 'config.yaml')
        if os.path.exists(saved_config):
            config_path = saved_config
            print(f"[Auto] Using training config: {saved_config}")

    # 加载配置
    config = load_config(config_path)

    # ====== 自适应科学配置：根据 dataset_selection 自动设置 ======
    if not validate_dataset_selection(config):
        print("ERROR: dataset_selection 配置无效，请确保恰好有一个要素为 true")
        return
    config = apply_element_settings(config)

    # 设置日志
    logger = setup_logger(
        name="HyperGKAN-Predict",
        level="INFO",
        console=True,
        file=False
    )

    logger.info("=" * 50)
    logger.info("HyperGKAN Prediction")
    logger.info("=" * 50)
    logger.info(f"Element: {config['meta']['element']}")

    # 如果未指定checkpoint，自动查找最新的
    if checkpoint_path is None:
        logger.info(f"{YELLOW}未指定checkpoint，自动查找最新训练结果...{RESET}")
        checkpoint_path, training_dir = find_latest_checkpoint()

        if checkpoint_path is None:
            logger.error("未找到任何checkpoint！请先训练模型或使用 --checkpoint 指定路径")
            return

        logger.info(f"{GREEN}✓ 找到最新checkpoint: {checkpoint_path}{RESET}")

        # 如果未指定输出目录，使用训练目录
        if output_dir is None and training_dir is not None:
            output_dir = training_dir
            logger.info(f"{GREEN}✓ 输出到训练目录: {output_dir}{RESET}")
    else:
        # 指定了checkpoint，使用推断的训练目录
        if output_dir is None:
            if training_dir is not None:
                output_dir = training_dir
                logger.info(f"{GREEN}✓ 输出到训练目录: {output_dir}{RESET}")
            else:
                # 使用绝对路径构建输出目录
                base_dir = config['output']['base_dir']
                if not os.path.isabs(base_dir):
                    base_dir = os.path.join(SCRIPT_DIR, base_dir)
                output_dir = os.path.join(
                    base_dir,
                    f"predictions_{config['meta']['element']}"
                )

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {output_dir}")

    # 设备
    device = config['meta']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    logger.info(f"Device: {device}")

    # 加载数据
    logger.info("=" * 50)
    logger.info("Loading data...")

    # 获取context特征掩码（按顺序：经度,纬度,海拔,年,月,日,时,区域）
    context_features = config['data'].get('context_features', {})
    context_feature_mask = [
        context_features.get('use_longitude', True),
        context_features.get('use_latitude', True),
        context_features.get('use_altitude', True),
        context_features.get('use_year', True),
        context_features.get('use_month', True),
        context_features.get('use_day', True),
        context_features.get('use_hour', True),
        context_features.get('use_region', True),
    ]
    logger.info(f"Context feature mask: {context_feature_mask} ({sum(context_feature_mask)} features enabled)")

    train_data = load_pkl_data(
        config['data']['train_path'],
        use_context=config['data']['use_context'],
        context_dim=config['data']['context_dim'],
        use_dim4=config['data']['use_dim4'],
        context_feature_mask=context_feature_mask
    )

    test_data = load_pkl_data(
        config['data']['test_path'],
        use_context=config['data']['use_context'],
        context_dim=config['data']['context_dim'],
        use_dim4=config['data']['use_dim4'],
        context_feature_mask=context_feature_mask
    )

    position = load_position_data(config['data']['position_path'])

    # 站点采样（如果配置了）
    num_stations = config['data'].get('num_stations', None)
    seed = config['meta'].get('seed', 42)
    original_num_stations = position.shape[0]

    logger.info(f"Station sampling check: num_stations={num_stations}, original_num_stations={original_num_stations}")

    if num_stations is not None and num_stations < original_num_stations:
        logger.info("=" * 50)
        logger.info(f"Station sampling: {original_num_stations} -> {num_stations}")

        # 设置随机种子（与训练时保持一致）
        np.random.seed(seed)

        # 随机选择站点索引
        selected_indices = np.random.choice(
            original_num_stations,
            num_stations,
            replace=False
        )
        selected_indices = np.sort(selected_indices)

        logger.info(f"  Selected station index range: {selected_indices[0]} ~ {selected_indices[-1]}")

        # 对test_data进行采样
        def sample_data(data_dict):
            """对单个数据集进行站点采样"""
            sampled = {}
            for key, value in data_dict.items():
                if key in ['x', 'y', 'context']:
                    if value is not None:
                        if value.ndim == 4:
                            sampled[key] = value[:, :, selected_indices, :]
                        elif value.ndim == 3:
                            sampled[key] = value[:, selected_indices, :]
                        else:
                            sampled[key] = value
                        logger.info(f"  {key}: {value.shape} -> {sampled[key].shape}")
                    else:
                        sampled[key] = None
                else:
                    sampled[key] = value
            return sampled

        test_data = sample_data(test_data)

        # 对position进行采样
        position = position[selected_indices]
        logger.info(f"  Position: {(original_num_stations, 2)} -> {position.shape}")

        logger.info(f"Station sampling completed")
    else:
        if num_stations is None:
            logger.info(f"Station sampling not configured, using all {original_num_stations} stations")
        elif num_stations >= original_num_stations:
            logger.info(f"Station num_stations ({num_stations}) >= original ({original_num_stations}), no sampling needed")

    # ====== 加载预处理器并对测试数据做标准化 ======
    # 【关键修复】必须在创建DataLoader之前对数据做标准化，
    # 因为模型是在标准化数据上训练的
    logger.info("=" * 50)
    logger.info("Loading preprocessor and normalizing test data...")

    # 预处理器在 output_dir 下（而非 checkpoints 子目录下）
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    preprocessor = None

    if os.path.exists(preprocessor_path):
        preprocessor = DataPreprocessor.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")

        # 对测试数据做标准化（与训练时一致）
        test_data = preprocessor.transform(test_data)
        logger.info("Test data normalized successfully")
        logger.info(f"  test_data['x'] range: [{test_data['x'].min():.4f}, {test_data['x'].max():.4f}]")
        if test_data.get('context') is not None:
            logger.info(f"  test_data['context'] range: [{test_data['context'].min():.4f}, {test_data['context'].max():.4f}]")
    else:
        logger.warning(f"Preprocessor not found at {preprocessor_path}")
        logger.warning("Test data will NOT be normalized - predictions may be incorrect!")

    # 构建超图
    logger.info("=" * 50)
    logger.info("Building hypergraphs...")

    cache_dir = config['graph']['cache_dir']
    element = config['meta']['element']

    # 使用采样后的站点数（如果配置了采样）
    num_stations = config['data'].get('num_stations', position.shape[0])

    # 邻域超图缓存路径
    top_k_nei = config['graph']['neighbourhood']['top_k']
    method_nei = config['graph']['neighbourhood']['method']
    use_geodesic = config['graph']['neighbourhood']['use_geodesic']
    weight_decay = config['graph']['neighbourhood']['weight_decay']
    station_suffix = f"_N{num_stations}" if num_stations else ""
    nei_cache_path = os.path.join(
        cache_dir,
        f"{element}_nei_K{top_k_nei}_{method_nei}_geo{use_geodesic}_decay{weight_decay}{station_suffix}.npz"
    )

    H_nei, W_nei = build_neighbourhood_hypergraph(
        positions=position,
        top_k=top_k_nei,
        method=method_nei,
        use_geodesic=use_geodesic,
        weight_decay=weight_decay,
        cache_path=nei_cache_path
    )

    # 语义超图缓存路径
    top_k_sem = config['graph']['semantic']['top_k']
    similarity_sem = config['graph']['semantic']['similarity']
    input_window_sem = config['graph']['semantic']['input_window']
    normalize_sem = config['graph']['semantic']['normalize_features']
    sem_cache_path = os.path.join(
        cache_dir,
        f"{element}_sem_K{top_k_sem}_{similarity_sem}_win{input_window_sem}_norm{normalize_sem}{station_suffix}.npz"
    )

    H_sem, W_sem = build_semantic_hypergraph(
        train_data=train_data['x'],
        top_k=top_k_sem,
        similarity=similarity_sem,
        input_window=input_window_sem,
        normalize_features=normalize_sem,
        cache_path=sem_cache_path
    )

    H_nei = H_nei.to(device)
    H_sem = H_sem.to(device)
    W_nei = W_nei.to(device) if W_nei is not None else None
    W_sem = W_sem.to(device) if W_sem is not None else None

    # 创建数据加载器
    logger.info("=" * 50)
    logger.info("Creating test loader...")

    # 计算正确的input_dim：如果使用context并拼接，需要加上context的维度
    use_context = config['data'].get('use_context', False)

    from src.data import SpatioTemporalDataset
    from torch.utils.data import DataLoader

    test_dataset = SpatioTemporalDataset(
        x=test_data['x'],
        y=test_data.get('y', None),
        context=test_data.get('context', None),
        input_window=config['data']['input_window'],
        output_window=config['data']['output_window'],
        stride=1,
        concat_context=use_context  # 与训练时保持一致
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # 创建模型
    logger.info("=" * 50)
    logger.info("Creating model...")

    # 先加载checkpoint获取实际的模型架构参数
    logger.info(f"Loading checkpoint to get model architecture: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

    # 从checkpoint获取保存的配置
    checkpoint_config = checkpoint_data.get('config', {})
    if checkpoint_config:
        logger.info("Using config saved in checkpoint")
        # 使用checkpoint中保存的配置
        model_config = checkpoint_config['model']
        graph_config = checkpoint_config['graph']
        data_config = checkpoint_config['data']

        # 计算正确的input_dim
        concat_context = data_config.get('use_context', False)
        input_dim = test_data['x'].shape[-1]
        if concat_context and test_data.get('context') is not None:
            input_dim += test_data['context'].shape[-1]
            logger.info(f"Context concatenated (per checkpoint config): input_dim = {test_data['x'].shape[-1]} + {test_data['context'].shape[-1]} = {input_dim}")

        conv_cfg = graph_config['conv']
        model = HyperGKAN(
            input_dim=input_dim,
            output_dim=model_config['output_projection']['output_dim'],
            d_model=model_config['input_projection']['d_model'],
            num_hypergkan_layers=model_config['hypergkan_layer']['num_layers'],
            hidden_channels=conv_cfg['hidden_channels'],
            gru_hidden_size=model_config['temporal']['hidden_size'],
            gru_num_layers=model_config['temporal']['num_layers'],
            use_kan=model_config['kan']['use_kan'],
            kan_grid_size=model_config['kan']['grid_size'],
            kan_spline_order=model_config['kan']['spline_order'],
            dropout=model_config['hypergkan_layer']['dropout'],
            fusion_method=model_config['hypergkan_layer']['fusion_method'],
            gru_type=model_config['temporal']['type'],
            float32_norm=conv_cfg.get('float32_norm', True),
            degree_clamp_min=conv_cfg.get('degree_clamp_min', 1e-6)
        )
    else:
        # 如果checkpoint中没有保存config，使用当前config
        logger.warning("No config found in checkpoint, using current config")
        input_dim = test_data['x'].shape[-1]
        if use_context and test_data.get('context') is not None:
            input_dim += test_data['context'].shape[-1]
            logger.info(f"Context will be concatenated: input_dim = {test_data['x'].shape[-1]} + {test_data['context'].shape[-1]} = {input_dim}")

        conv_cfg = config['graph']['conv']
        model = HyperGKAN(
            input_dim=input_dim,
            output_dim=config['model']['output_projection']['output_dim'],
            d_model=config['model']['input_projection']['d_model'],
            num_hypergkan_layers=config['model']['hypergkan_layer']['num_layers'],
            hidden_channels=conv_cfg['hidden_channels'],
            gru_hidden_size=config['model']['temporal']['hidden_size'],
            gru_num_layers=config['model']['temporal']['num_layers'],
            use_kan=config['model']['kan']['use_kan'],
            kan_grid_size=config['model']['kan']['grid_size'],
            kan_spline_order=config['model']['kan']['spline_order'],
            dropout=config['model']['hypergkan_layer']['dropout'],
            fusion_method=config['model']['hypergkan_layer']['fusion_method'],
            gru_type=config['model']['temporal']['type'],
            float32_norm=conv_cfg.get('float32_norm', True),
            degree_clamp_min=conv_cfg.get('degree_clamp_min', 1e-6)
        )

    # 加载模型权重
    logger.info(f"Loading model weights from checkpoint")
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)

    # 预测
    logger.info("=" * 50)
    inputs, predictions, targets = predict(
        model, test_loader, H_nei, H_sem, W_nei, W_sem, device, logger
    )

    # 反标准化预测结果和目标值（用于在原始尺度上评估指标）
    logger.info("=" * 50)
    logger.info("Denormalizing predictions and targets...")
    if preprocessor is not None and preprocessor.fitted:
        predictions = preprocessor.inverse_transform(predictions)
        targets = preprocessor.inverse_transform(targets)
        logger.info("Predictions and targets denormalized successfully")
        logger.info(f"  Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        logger.info(f"  Targets range: [{targets.min():.2f}, {targets.max():.2f}]")
    else:
        logger.warning("Preprocessor not available, using raw predictions for evaluation")

    # 评估
    overall_metrics, horizon_metrics, step_metrics = evaluate_predictions(
        predictions, targets, config, logger
    )

    # 基线对比
    comparison = compare_with_baselines(
        overall_metrics, horizon_metrics,
        config['meta']['element'], logger
    )

    # 保存结果
    save_results(
        inputs, predictions, targets,
        overall_metrics, horizon_metrics,
        output_dir, config, logger, comparison,
        step_metrics=step_metrics
    )

    logger.info("=" * 50)
    logger.info(f"{GREEN}✓ Prediction completed!{RESET}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HyperGKAN Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动使用最新训练结果
  python predict.py

  # 指定checkpoint
  python predict.py --checkpoint outputs/20260127_153303_Temperature/checkpoints/best_model.pt

  # 指定输出目录
  python predict.py --checkpoint model.pt --output custom_output_dir

  # 指定GPU
  python predict.py --checkpoint model.pt --gpu 1
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (默认: 自动查找最新训练结果)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (默认: checkpoint对应的训练目录)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device number to use (overrides config)'
    )

    args = parser.parse_args()
    main(args)
