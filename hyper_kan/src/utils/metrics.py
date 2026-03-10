"""
评估指标
"""
import torch
import numpy as np
from typing import Dict, List


def MAE(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Mean Absolute Error"""
    if mask is not None:
        error = torch.abs(pred - target) * mask
        return error.sum() / mask.sum()
    else:
        return torch.mean(torch.abs(pred - target))


def RMSE(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Root Mean Squared Error"""
    if mask is not None:
        error = ((pred - target) ** 2) * mask
        return torch.sqrt(error.sum() / mask.sum())
    else:
        return torch.sqrt(torch.mean((pred - target) ** 2))


def MAPE(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, epsilon: float = 1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error"""
    if mask is not None:
        error = torch.abs((pred - target) / (torch.abs(target) + epsilon)) * mask
        return (error.sum() / mask.sum()) * 100.0
    else:
        return torch.mean(torch.abs((pred - target) / (torch.abs(target) + epsilon))) * 100.0


def compute_metrics(pred: torch.Tensor, 
                   target: torch.Tensor,
                   metrics: List[str] = ['mae', 'rmse', 'mape'],
                   mask: torch.Tensor = None) -> Dict[str, float]:
    """
    计算多个评估指标
    
    Args:
        pred: 预测值 (任意shape)
        target: 真实值 (与pred相同shape)
        metrics: 指标列表
        mask: 掩码 (可选)
    
    Returns:
        指标字典 {'mae': ..., 'rmse': ..., ...}
    """
    results = {}
    
    for metric in metrics:
        metric_lower = metric.lower()
        
        if metric_lower == 'mae':
            results['mae'] = MAE(pred, target, mask).item()
        
        elif metric_lower == 'rmse':
            results['rmse'] = RMSE(pred, target, mask).item()
        
        elif metric_lower == 'mape':
            results['mape'] = MAPE(pred, target, mask).item()
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return results


def evaluate_by_horizon(pred: torch.Tensor,
                        target: torch.Tensor,
                        horizons: List[int] = [3, 6, 12]) -> Dict[int, Dict[str, float]]:
    """
    按预测步长评估
    
    Args:
        pred: (B, T, N, F)
        target: (B, T, N, F)
        horizons: 评估的时间步列表
    
    Returns:
        {horizon: {'mae': ..., 'rmse': ...}, ...}
    """
    results = {}
    
    for h in horizons:
        if h > pred.shape[1]:
            continue
        
        pred_h = pred[:, :h, :, :]
        target_h = target[:, :h, :, :]
        
        metrics = compute_metrics(pred_h, target_h)
        results[h] = metrics
    
    return results
