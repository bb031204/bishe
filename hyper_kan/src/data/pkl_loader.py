"""
数据加载器 - 支持多种PKL格式
"""
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_pkl_data(file_path: str, 
                  use_context: bool = True,
                  context_dim: int = 4,
                  use_dim4: bool = False,
                  context_feature_mask: Optional[List[bool]] = None) -> Dict[str, np.ndarray]:
    """
    加载PKL格式的气象数据
    
    Args:
        file_path: PKL文件路径
        use_context: 是否使用context特征
        context_dim: context维度
        use_dim4: 是否使用第4维特征
        context_feature_mask: 8个context特征的选择掩码 [经度,纬度,海拔,年,月,日,时,区域]
    
    Returns:
        包含 'x', 'y', 'context', 'position' 的字典
    """
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检测数据格式
    if isinstance(data, dict):
        # 格式1: {'x': ..., 'y': ..., 'context': ..., 'position': ...}
        x = data.get('x', None)
        y = data.get('y', None)
        context = data.get('context', None)
        position = data.get('position', None)
        
    elif isinstance(data, (list, tuple)):
        # 格式2: (x, y, context) 或 (x, y)
        if len(data) == 3:
            x, y, context = data
            position = None
        elif len(data) == 2:
            x, y = data
            context = None
            position = None
        else:
            raise ValueError(f"Unexpected tuple length: {len(data)}")
    
    elif isinstance(data, np.ndarray):
        # 格式3: 纯numpy数组
        x = data
        y = None
        context = None
        position = None
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # 数据验证
    if x is None:
        raise ValueError("Data must contain 'x' field")
    
    # 转换为numpy数组
    x = np.array(x)
    if y is not None:
        y = np.array(y)
    if context is not None:
        context = np.array(context)
    if position is not None:
        position = np.array(position)
    
    # 处理context
    if use_context and context is not None:
        if context.ndim == 2:
            # (N, C) -> 扩展到每个时间步
            # 假设x的shape为 (T, N, F)
            T = x.shape[0]
            context = np.tile(context[np.newaxis, :, :], (T, 1, 1))
        
        # 应用context特征掩码（选择要使用的特征）
        if context_feature_mask is not None and len(context_feature_mask) > 0:
            # 确保掩码长度不超过context维度
            mask_len = min(len(context_feature_mask), context.shape[-1])
            selected_indices = [i for i in range(mask_len) if context_feature_mask[i]]
            
            if len(selected_indices) > 0:
                context = context[..., selected_indices]
                logger.info(f"Context features selected: {len(selected_indices)}/{mask_len} features")
            else:
                logger.warning("No context features selected by mask, disabling context")
                context = None
        
        # 限制context维度（如果未使用掩码）
        elif context.shape[-1] > context_dim:
            context = context[..., :context_dim]
            logger.info(f"Context dimension limited to {context_dim}")
    
    # 处理第4维特征
    if not use_dim4 and x.ndim >= 3 and x.shape[-1] > 3:
        x = x[..., :3]
        if y is not None and y.ndim >= 3 and y.shape[-1] > 3:
            y = y[..., :3]
        logger.info("Using first 3 dimensions only (dim4 disabled)")
    
    # 数据形状信息
    logger.info(f"Data loaded - x: {x.shape}")
    if y is not None:
        logger.info(f"              y: {y.shape}")
    if context is not None:
        logger.info(f"              context: {context.shape}")
    if position is not None:
        logger.info(f"              position: {position.shape}")
    
    result = {
        'x': x,
        'y': y,
        'context': context if use_context else None,
        'position': position
    }
    
    return result


def load_position_data(file_path: str) -> np.ndarray:
    """
    加载站点位置信息
    
    Args:
        file_path: position.pkl文件路径
    
    Returns:
        位置数组 (N, 2) - [latitude, longitude]，如果加载失败返回None
    """
    try:
        logger.info(f"Loading position data from {file_path}")
        
        with open(file_path, 'rb') as f:
            position = pickle.load(f)
        
        # 如果是字典，尝试提取位置数据
        if isinstance(position, dict):
            if 'lonlat' in position:
                logger.info("Position data is dict with 'lonlat' key, extracting...")
                position = position['lonlat']
            elif 'position' in position:
                logger.info("Position data is dict with 'position' key, extracting...")
                position = position['position']
            else:
                logger.error(f"Position dict has unexpected keys: {position.keys()}")
                return None
        
        # 尝试转换为numpy数组
        position = np.array(position)
        
        # 验证形状
        if position.ndim == 0:
            logger.error(f"Position data is scalar")
            return None
        
        if position.ndim == 1:
            logger.warning(f"Position data is 1D: {position.shape}, trying to reshape...")
            if len(position) % 2 == 0:
                position = position.reshape(-1, 2)
                logger.info(f"Reshaped to: {position.shape}")
            else:
                logger.error(f"Cannot reshape 1D array with odd length: {len(position)}")
                return None
        
        if position.ndim != 2:
            logger.warning(f"Unexpected position dimensions: {position.ndim}D, expected 2D")
            return None
        
        if position.shape[1] != 2:
            logger.warning(f"Unexpected position columns: {position.shape[1]}, expected 2 (lat, lon)")
            # 如果有多列，尝试只取前两列
            if position.shape[1] > 2:
                logger.info(f"Taking first 2 columns from {position.shape[1]} columns")
                position = position[:, :2]
            else:
                return None
        
        logger.info(f"Position data loaded: {position.shape[0]} stations, shape: {position.shape}")
        return position
    
    except FileNotFoundError:
        logger.error(f"Position file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading position data: {e}")
        return None


def save_pkl_data(data: Any, file_path: str):
    """
    保存数据为PKL格式
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
    """
    logger.info(f"Saving data to {file_path}")
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info("Data saved successfully")
