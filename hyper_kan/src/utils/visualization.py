"""
可视化工具
"""
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)

# 全局风格设置
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "Loss Curve"
):
    """
    绘制损失曲线
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MAE)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss curve saved to {save_path}")


def _select_representative_samples(
    pred: np.ndarray,
    target: np.ndarray,
    num_samples: int = 4
) -> List[int]:
    """
    智能选择有代表性的样本（最佳、中位数、较差、趋势变化明显的）
    
    Args:
        pred: (B, T, N, F)
        target: (B, T, N, F)
        num_samples: 要选择的样本数
    
    Returns:
        选中的样本索引列表
    """
    B = pred.shape[0]
    # 计算每个样本的MAE
    sample_mae = np.mean(np.abs(pred - target), axis=(1, 2, 3))  # (B,)
    
    sorted_indices = np.argsort(sample_mae)
    
    selected = set()
    # 最佳样本
    selected.add(sorted_indices[0])
    # 次佳样本
    if B > 1:
        selected.add(sorted_indices[max(1, B // 10)])
    # 中位数样本
    if B > 2:
        selected.add(sorted_indices[B // 2])
    # 较差但非最差（90%分位）
    if B > 3:
        selected.add(sorted_indices[min(B - 1, int(B * 0.9))])
    
    # 如果还不够，补充
    idx = 0
    while len(selected) < num_samples and idx < B:
        selected.add(sorted_indices[idx])
        idx += 1
    
    result = sorted(list(selected))[:num_samples]
    return result


def _select_diverse_stations(
    pred: np.ndarray,
    target: np.ndarray,
    num_stations: int = 4
) -> List[int]:
    """
    选择多样化的站点（表现好的、一般的、差的各挑一些）
    
    Args:
        pred: (B, T, N, F)
        target: (B, T, N, F)
        num_stations: 要选的站点数
    
    Returns:
        选中的站点索引列表
    """
    N = pred.shape[2]
    # 计算每个站点的MAE
    station_mae = np.mean(np.abs(pred - target), axis=(0, 1, 3))  # (N,)
    
    sorted_indices = np.argsort(station_mae)
    
    selected = set()
    # 最佳站点
    selected.add(sorted_indices[0])
    # 次佳
    if N > 1:
        selected.add(sorted_indices[max(1, N // 10)])
    # 中位数
    if N > 2:
        selected.add(sorted_indices[N // 2])
    # 较差
    if N > 3:
        selected.add(sorted_indices[min(N - 1, int(N * 0.75))])
    
    idx = 0
    while len(selected) < num_stations and idx < N:
        selected.add(sorted_indices[idx])
        idx += 1
    
    result = sorted(list(selected))[:num_stations]
    return result


def _kelvin_to_celsius(values: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    如果数据看起来是开尔文温度，转换为摄氏度
    
    Returns:
        (转换后的值, 是否进行了转换)
    """
    mean_val = np.mean(values)
    if 180 < mean_val < 350:  # 典型开尔文温度范围
        return values - 273.15, True
    return values, False


def plot_predictions(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    num_samples: int = 4,
    num_stations: int = 4,
    horizon_steps: Optional[List[int]] = None,
    element: str = "Temperature"
):
    """
    绘制预测结果（增强版）
    
    - 智能选择有代表性的样本和站点
    - 自动开尔文转摄氏度
    - 每个子图标注MAE
    - 误差阴影区域
    - 更美观的配色和布局
    
    Args:
        pred: 预测值 (B, T, N, F)
        target: 真实值 (B, T, N, F)
        save_path: 保存路径
        num_samples: 绘制的样本数量
        num_stations: 每个样本绘制的站点数量
        horizon_steps: 要标注的预测步长
        element: 气象要素名称
    """
    B, T, N, F = pred.shape
    
    num_samples = min(num_samples, B)
    num_stations = min(num_stations, N)
    
    if horizon_steps is None:
        horizon_steps = [3, 6, 12] if T >= 12 else [T // 3, T // 2, T]
    
    # 智能选择样本和站点
    sample_indices = _select_representative_samples(pred, target, num_samples)
    station_indices = _select_diverse_stations(pred, target, num_stations)
    
    # 检查是否需要开尔文转摄氏度
    pred_plot, converted = _kelvin_to_celsius(pred)
    target_plot, _ = _kelvin_to_celsius(target)
    
    # 根据气象要素设置单位
    ELEMENT_UNITS = {
        'Temperature': '°C',
        'Cloud': 'fraction',
        'Humidity': '%',
        'Wind': 'm/s',
    }
    if converted:
        unit = "°C"
    else:
        unit = ELEMENT_UNITS.get(element, "")
    
    # 计算全局整体MAE（转换后的尺度）
    overall_mae = np.mean(np.abs(pred_plot - target_plot))
    
    # ============ 图1: 样本预测曲线（主图） ============
    fig, axes = plt.subplots(
        num_samples, num_stations,
        figsize=(5.5 * num_stations, 4 * num_samples)
    )
    
    if num_samples == 1 and num_stations == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = axes.reshape(1, -1)
    elif num_stations == 1:
        axes = axes.reshape(-1, 1)
    
    # 颜色方案
    color_gt = '#2166AC'       # 深蓝
    color_pred = '#D6604D'     # 暗红
    color_fill = '#FDDBC7'    # 浅橙（误差填充）
    color_horizon = '#4DAF4A'  # 绿色（步长线）
    
    for row_idx, si in enumerate(sample_indices):
        for col_idx, sj in enumerate(station_indices):
            ax = axes[row_idx, col_idx]
            
            pred_series = pred_plot[si, :, sj, 0]
            target_series = target_plot[si, :, sj, 0]
            
            time_steps = np.arange(1, T + 1)
            
            # 误差填充
            ax.fill_between(
                time_steps, target_series, pred_series,
                alpha=0.2, color=color_fill, label='Error'
            )
            
            # 真实值和预测值
            ax.plot(time_steps, target_series, color=color_gt,
                    linewidth=2.2, marker='o', markersize=3.5,
                    label='Ground Truth', zorder=3)
            ax.plot(time_steps, pred_series, color=color_pred,
                    linewidth=2.2, marker='s', markersize=3,
                    linestyle='--', label='Prediction', zorder=3)
            
            # 标注关键步长
            for h in horizon_steps:
                if h <= T:
                    ax.axvline(x=h, color=color_horizon, linestyle=':', alpha=0.4, linewidth=1)
            
            # 计算子图MAE
            sub_mae = np.mean(np.abs(pred_series - target_series))
            
            ax.set_xlabel('Forecast Hour', fontsize=10)
            y_label = f'{element} ({unit})' if unit else element
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(
                f'Sample {si+1}, Station {sj+1}  |  MAE={sub_mae:.2f}{unit}',
                fontsize=11, fontweight='bold'
            )
            ax.legend(fontsize=8, loc='best', framealpha=0.8)
            ax.grid(True, alpha=0.2, linestyle='-')
            ax.set_xticks(time_steps)
            
            # y轴留白
            y_min = min(pred_series.min(), target_series.min())
            y_max = max(pred_series.max(), target_series.max())
            margin = max((y_max - y_min) * 0.15, 0.5)
            ax.set_ylim(y_min - margin, y_max + margin)
    
    fig.suptitle(
        f'HyperGKAN {element} Prediction  (Overall MAE={overall_mae:.3f}{unit})',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Predictions plot saved to {save_path}")
    
    # ============ 图2: 综合分析图 ============
    analysis_path = save_path.replace('.png', '_analysis.png')
    fig2 = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    
    # --- (a) 散点图: Prediction vs Ground Truth ---
    ax1 = fig2.add_subplot(gs[0, 0])
    # 随机抽样避免绘图太慢
    flat_pred = pred_plot[:, :, :, 0].flatten()
    flat_target = target_plot[:, :, :, 0].flatten()
    n_points = len(flat_pred)
    if n_points > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_points, 50000, replace=False)
        flat_pred_s = flat_pred[idx]
        flat_target_s = flat_target[idx]
    else:
        flat_pred_s = flat_pred
        flat_target_s = flat_target
    
    ax1.scatter(flat_target_s, flat_pred_s, alpha=0.08, s=3, c='#4393C3', rasterized=True)
    vmin = min(flat_target_s.min(), flat_pred_s.min())
    vmax = max(flat_target_s.max(), flat_pred_s.max())
    ax1.plot([vmin, vmax], [vmin, vmax], 'r-', linewidth=1.5, label='y = x (Perfect)')
    ax1.set_xlabel(f'Ground Truth ({unit})', fontsize=11)
    ax1.set_ylabel(f'Prediction ({unit})', fontsize=11)
    ax1.set_title('(a) Prediction vs Ground Truth', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal', adjustable='box')
    
    # --- (b) 误差分布直方图 ---
    ax2 = fig2.add_subplot(gs[0, 1])
    errors = flat_pred - flat_target
    if n_points > 50000:
        errors_s = flat_pred_s - flat_target_s
    else:
        errors_s = errors
    ax2.hist(errors_s, bins=80, color='#92C5DE', edgecolor='#4393C3',
             alpha=0.8, density=True)
    ax2.axvline(x=0, color='red', linewidth=1.5, linestyle='--')
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    ax2.axvline(x=mean_err, color='#D6604D', linewidth=1.2,
                linestyle='-', label=f'Mean={mean_err:.3f}')
    ax2.set_xlabel(f'Prediction Error ({unit})', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'(b) Error Distribution (std={std_err:.3f})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    # --- (c) 各预测步长MAE ---
    ax3 = fig2.add_subplot(gs[0, 2])
    horizon_mae = []
    for t in range(T):
        h_mae = np.mean(np.abs(pred_plot[:, t, :, 0] - target_plot[:, t, :, 0]))
        horizon_mae.append(h_mae)
    
    bars = ax3.bar(range(1, T + 1), horizon_mae,
                   color=['#92C5DE' if t + 1 not in horizon_steps else '#D6604D' for t in range(T)],
                   edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Forecast Step (hour)', fontsize=11)
    ax3.set_ylabel(f'MAE ({unit})', fontsize=11)
    ax3.set_title('(c) MAE by Forecast Horizon', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(1, T + 1))
    ax3.grid(True, alpha=0.2, axis='y')
    # 在关键步长的bar上标数值
    for t in range(T):
        if (t + 1) in horizon_steps:
            ax3.text(t + 1, horizon_mae[t] + 0.02, f'{horizon_mae[t]:.3f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # --- (d) 站点级MAE箱线图 ---
    ax4 = fig2.add_subplot(gs[1, 0])
    station_maes = np.mean(np.abs(pred_plot[:, :, :, 0] - target_plot[:, :, :, 0]),
                           axis=(0, 1))  # (N,)
    bp = ax4.boxplot(station_maes, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='#92C5DE', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax4.set_ylabel(f'MAE ({unit})', fontsize=11)
    ax4.set_title(f'(d) Station-level MAE Distribution (N={N})', fontsize=12, fontweight='bold')
    ax4.set_xticklabels(['All Stations'])
    ax4.grid(True, alpha=0.2, axis='y')
    q25, median, q75 = np.percentile(station_maes, [25, 50, 75])
    ax4.text(1.25, median, f'Median: {median:.3f}\nQ25: {q25:.3f}\nQ75: {q75:.3f}',
             fontsize=9, va='center')
    
    # --- (e) 最佳样本+站点展示 & (f) 最差样本+站点展示 ---
    # 直接在所有 (sample, station) 组合中找 MAE 最低和最高的
    pair_mae = np.mean(np.abs(pred_plot[:, :, :, 0] - target_plot[:, :, :, 0]),
                       axis=1)  # (B, N)
    best_flat_idx = np.argmin(pair_mae)
    worst_flat_idx = np.argmax(pair_mae)
    best_si, best_sj = np.unravel_index(best_flat_idx, pair_mae.shape)
    worst_si, worst_sj = np.unravel_index(worst_flat_idx, pair_mae.shape)
    
    time_steps = np.arange(1, T + 1)
    
    # (e) Best Case
    ax5 = fig2.add_subplot(gs[1, 1])
    pred_best = pred_plot[best_si, :, best_sj, 0]
    target_best = target_plot[best_si, :, best_sj, 0]
    
    ax5.fill_between(time_steps, target_best, pred_best, alpha=0.2, color=color_fill)
    ax5.plot(time_steps, target_best, color=color_gt, linewidth=2.5,
             marker='o', markersize=5, label='Ground Truth', zorder=3)
    ax5.plot(time_steps, pred_best, color=color_pred, linewidth=2.5,
             marker='s', markersize=4, linestyle='--', label='Prediction', zorder=3)
    best_mae = pair_mae[best_si, best_sj]
    ax5.set_xlabel('Forecast Hour', fontsize=11)
    ax5.set_ylabel(f'{element} ({unit})', fontsize=11)
    ax5.set_title(f'(e) Best Case (MAE={best_mae:.3f}{unit})', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.2)
    ax5.set_xticks(time_steps)
    
    # (f) Hard Case
    ax6 = fig2.add_subplot(gs[1, 2])
    pred_worst = pred_plot[worst_si, :, worst_sj, 0]
    target_worst = target_plot[worst_si, :, worst_sj, 0]
    
    ax6.fill_between(time_steps, target_worst, pred_worst, alpha=0.2, color=color_fill)
    ax6.plot(time_steps, target_worst, color=color_gt, linewidth=2.5,
             marker='o', markersize=5, label='Ground Truth', zorder=3)
    ax6.plot(time_steps, pred_worst, color=color_pred, linewidth=2.5,
             marker='s', markersize=4, linestyle='--', label='Prediction', zorder=3)
    worst_mae = pair_mae[worst_si, worst_sj]
    ax6.set_xlabel('Forecast Hour', fontsize=11)
    ax6.set_ylabel(f'{element} ({unit})', fontsize=11)
    ax6.set_title(f'(f) Hard Case (MAE={worst_mae:.3f}{unit})', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.2)
    ax6.set_xticks(time_steps)
    
    fig2.suptitle(
        f'HyperGKAN {element} Prediction Analysis  |  Overall MAE={overall_mae:.3f}{unit}  '
        f'RMSE={np.sqrt(np.mean((pred_plot - target_plot) ** 2)):.3f}{unit}',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis plot saved to {analysis_path}")


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Metrics Comparison"
):
    """
    绘制多个模型的指标对比
    """
    models = list(metrics_dict.keys())
    metric_names = list(list(metrics_dict.values())[0].keys())
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model][m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Metrics comparison saved to {save_path}")
