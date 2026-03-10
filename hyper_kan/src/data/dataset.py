"""
PyTorch Dataset for Spatio-Temporal Data
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SpatioTemporalDataset(Dataset):
    """
    时空数据集
    
    支持滑动窗口构建输入输出序列对
    支持将context特征拼接到x中
    """
    
    def __init__(self,
                 x: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 context: Optional[np.ndarray] = None,
                 input_window: int = 12,
                 output_window: int = 12,
                 stride: int = 1,
                 concat_context: bool = True):
        """
        Args:
            x: 输入特征 (T, N, F) 或 预构建的 (S, Tin, N, F)
            y: 目标值 (T, N, F) 或 预构建的 (S, Tout, N, F), 如果为None则从x中生成
            context: 上下文特征 (T, N, C) 或 (S, Tin, N, C)
            input_window: 输入时间窗口
            output_window: 输出时间窗口
            stride: 滑动窗口步长
            concat_context: 是否将context拼接到x中（默认True）
        """
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.concat_context = concat_context
        
        # 检查是否已经是预构建的样本格式
        if x.ndim == 4:  # (S, Tin, N, F)
            self.is_prebuilt = True
            
            # 如果有context且需要拼接，先拼接再转换
            if context is not None and concat_context:
                # 拼接context到x
                x = np.concatenate([x, context], axis=-1)
                logger.info(f"Context concatenated to x: {x.shape}")
            
            self.samples_x = torch.from_numpy(x).float()
            
            if y is not None:
                self.samples_y = torch.from_numpy(y).float()
            else:
                # 如果没有y，使用x的最后output_window步作为y
                # 注意：y不应包含context特征，所以需要分离
                if context is not None and concat_context:
                    # 从拼接后的x中分离出原始特征部分
                    original_feat_dim = x.shape[-1] - context.shape[-1]
                    self.samples_y = self.samples_x[:, -output_window:, :, :original_feat_dim]
                else:
                    self.samples_y = self.samples_x[:, -output_window:, :, :]
            
            self.samples_context = None  # context已经拼接到x中，不再单独保存
            
            self.num_samples = self.samples_x.shape[0]
            
            logger.info(f"Using pre-built samples: {self.num_samples} samples")
        
        else:  # (T, N, F) - 需要构建滑动窗口
            self.is_prebuilt = False
            
            # 如果有context且需要拼接，先拼接
            if context is not None and concat_context:
                self.x = np.concatenate([x, context], axis=-1)
                self.original_feat_dim = x.shape[-1]
                logger.info(f"Context concatenated to x: original_dim={x.shape[-1]}, "
                           f"context_dim={context.shape[-1]}, total_dim={self.x.shape[-1]}")
            else:
                self.x = x
                self.original_feat_dim = x.shape[-1]
            
            self.y = y if y is not None else x  # 如果没有y，使用原始x作为y（不含context）
            self.context = None  # context已经拼接，不再单独保存
            
            # 计算可以构建的样本数量
            total_window = input_window + output_window
            T = self.x.shape[0]
            self.num_samples = max(0, (T - total_window) // stride + 1)
            
            if self.num_samples == 0:
                raise ValueError(
                    f"Insufficient time steps: T={T}, required={total_window}, "
                    f"got {self.num_samples} samples"
                )
            
            logger.info(f"Building sliding windows: {self.num_samples} samples "
                       f"(input={input_window}, output={output_window}, stride={stride})")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_prebuilt:
            # 预构建样本
            sample = {
                'x': self.samples_x[idx],  # (Tin, N, F+C) - 已拼接context
                'y': self.samples_y[idx],  # (Tout, N, F) - 不含context
            }
            # context已经拼接到x中，不再单独返回
        
        else:
            # 滑动窗口构建
            start_idx = idx * self.stride
            end_input = start_idx + self.input_window
            end_output = end_input + self.output_window
            
            x_window = self.x[start_idx:end_input]  # (Tin, N, F+C) - 已拼接context
            y_window = self.y[end_input:end_output]  # (Tout, N, F) - 不含context
            
            sample = {
                'x': torch.from_numpy(x_window).float(),
                'y': torch.from_numpy(y_window).float(),
            }
            # context已经拼接到x中，不再单独返回
        
        return sample


def create_data_loaders(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    input_window: int,
    output_window: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
    stride: int = 1,
    concat_context: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试的DataLoader
    
    Args:
        train_data: 训练数据字典 {'x': ..., 'y': ..., 'context': ...}
        val_data: 验证数据字典
        test_data: 测试数据字典
        input_window: 输入时间窗口
        output_window: 输出时间窗口
        batch_size: 批次大小
        num_workers: 数据加载线程数
        shuffle_train: 是否打乱训练集
        stride: 滑动窗口步长
        concat_context: 是否将context拼接到x中
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    train_dataset = SpatioTemporalDataset(
        x=train_data['x'],
        y=train_data.get('y', None),
        context=train_data.get('context', None),
        input_window=input_window,
        output_window=output_window,
        stride=stride,
        concat_context=concat_context
    )
    
    val_dataset = SpatioTemporalDataset(
        x=val_data['x'],
        y=val_data.get('y', None),
        context=val_data.get('context', None),
        input_window=input_window,
        output_window=output_window,
        stride=stride,
        concat_context=concat_context
    )
    
    test_dataset = SpatioTemporalDataset(
        x=test_data['x'],
        y=test_data.get('y', None),
        context=test_data.get('context', None),
        input_window=input_window,
        output_window=output_window,
        stride=stride,
        concat_context=concat_context
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
