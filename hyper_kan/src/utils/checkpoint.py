"""
Checkpoint管理
"""
import torch
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False
):
    """
    保存checkpoint

    Args:
        state: 状态字典 (包含model, optimizer, epoch等)
        checkpoint_dir: checkpoint目录
        filename: 文件名 (会被保存为 last.pt)
        is_best: 是否是最佳模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存最新checkpoint (固定文件名 last.pt，覆盖之前的)
    last_path = os.path.join(checkpoint_dir, "last.pt")
    torch.save(state, last_path)
    logger.info(f"Checkpoint saved: {last_path}")

    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_path)
        logger.info(f"✨ Best model saved: {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    加载checkpoint
    
    Args:
        checkpoint_path: checkpoint路径
        model: 模型
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        device: 设备
    
    Returns:
        包含epoch, metrics等信息的字典
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded: epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    获取最新的checkpoint路径
    
    Args:
        checkpoint_dir: checkpoint目录
    
    Returns:
        最新checkpoint路径，如果不存在返回None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有checkpoint文件
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pt') or f.endswith('.pth')
    ]
    
    if len(checkpoints) == 0:
        return None
    
    # 优先返回best_model
    if 'best_model.pt' in checkpoints:
        return os.path.join(checkpoint_dir, 'best_model.pt')
    
    # 否则返回最新修改的文件
    checkpoints = [os.path.join(checkpoint_dir, f) for f in checkpoints]
    latest = max(checkpoints, key=os.path.getmtime)
    
    return latest
