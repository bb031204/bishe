"""
数据预处理模块 - 标准化和温度转换
"""
import numpy as np
import pickle
import logging
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self,
                 kelvin_to_celsius: bool = False,
                 normalize: bool = True,
                 scaler_type: str = 'standard',
                 context_dim: int = 0):
        """
        Args:
            kelvin_to_celsius: 是否将开尔文转换为摄氏度
            normalize: 是否标准化数据
            scaler_type: 标准化类型 ('standard' 或 'minmax')
            context_dim: context特征维度（用于分离标准化）
        """
        self.kelvin_to_celsius = kelvin_to_celsius
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.context_dim = context_dim
        self.fitted = False

        # 标准化器（只用于气象特征，不用于context）
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        # context特征的独立标准化器
        if context_dim > 0:
            if scaler_type == 'standard':
                self.context_scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.context_scaler = MinMaxScaler()
        else:
            self.context_scaler = None

    def _kelvin_to_celsius_transform(self, data: np.ndarray) -> np.ndarray:
        """开尔文转摄氏度"""
        logger.info("Converting Kelvin to Celsius...")
        return data - 273.15

    def fit(self, train_data: Dict[str, np.ndarray]) -> 'DataPreprocessor':
        """
        在训练数据上拟合预处理器

        Args:
            train_data: 训练数据字典 {'x': ..., 'y': ..., 'context': ...}

        Returns:
            self
        """
        logger.info("=" * 50)
        logger.info("Fitting data preprocessor...")

        x = train_data['x']
        context = train_data.get('context', None)

        # 1. 温度转换 (如果需要) - 只处理气象特征
        if self.kelvin_to_celsius:
            logger.info(f"  Original temperature range: [{x.min():.2f}, {x.max():.2f}]")
            x = self._kelvin_to_celsius_transform(x)
            logger.info(f"  After K→C conversion: [{x.min():.2f}, {x.max():.2f}]")

        # 2. 标准化 (如果需要)
        if self.normalize:
            logger.info(f"  Fitting {self.scaler_type} scaler...")

            # 标准化气象特征
            original_shape = x.shape
            x_2d = x.reshape(-1, original_shape[-1])
            self.scaler.fit(x_2d)
            
            logger.info(f"  Weather features scaler fitted on {x_2d.shape[0]} samples")
            logger.info(f"    Mean: {self.scaler.mean_[0]:.4f}")
            if hasattr(self.scaler, 'scale_'):
                logger.info(f"    Std:  {self.scaler.scale_[0]:.4f}")
            
            # 如果有context，也为context拟合标准化器
            if context is not None and self.context_dim > 0 and self.context_scaler is not None:
                context_shape = context.shape
                context_2d = context.reshape(-1, context_shape[-1])
                self.context_scaler.fit(context_2d)
                
                logger.info(f"  Context features scaler fitted on {context_2d.shape[0]} samples")
                if hasattr(self.context_scaler, 'mean_'):
                    logger.info(f"    Mean (first): {self.context_scaler.mean_[0]:.4f}")

        self.fitted = True
        logger.info("✓ Preprocessor fitted")
        logger.info("=" * 50)

        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        应用预处理

        Args:
            data: 数据字典

        Returns:
            预处理后的数据
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted! Call fit() first.")

        result = {}

        # 处理 x (气象特征)
        if 'x' in data:
            x = data['x'].copy()

            # 1. 温度转换
            if self.kelvin_to_celsius:
                x = self._kelvin_to_celsius_transform(x)

            # 2. 标准化
            if self.normalize:
                original_shape = x.shape
                x_2d = x.reshape(-1, original_shape[-1])
                x_scaled = self.scaler.transform(x_2d)
                x = x_scaled.reshape(original_shape)

            result['x'] = x

        # 处理 y (气象特征)
        if 'y' in data:
            y = data['y'].copy()

            if self.kelvin_to_celsius:
                y = self._kelvin_to_celsius_transform(y)

            if self.normalize:
                original_shape = y.shape
                y_2d = y.reshape(-1, original_shape[-1])
                y_scaled = self.scaler.transform(y_2d)
                y = y_scaled.reshape(original_shape)

            result['y'] = y

        # 处理 context (单独标准化)
        if 'context' in data and data['context'] is not None:
            context = data['context'].copy()
            
            if self.normalize and self.context_scaler is not None:
                original_shape = context.shape
                context_2d = context.reshape(-1, original_shape[-1])
                context_scaled = self.context_scaler.transform(context_2d)
                context = context_scaled.reshape(original_shape)
            
            result['context'] = context

        # 复制其他字段
        for key in ['position']:
            if key in data:
                result[key] = data[key]

        return result

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反标准化（用于预测结果）
        注意：这里只处理气象特征，不处理context

        Args:
            data: 标准化后的数据（只含气象特征）

        Returns:
            原始尺度的数据
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted!")

        result = data.copy()

        # 1. 反标准化（只用气象特征的scaler）
        if self.normalize:
            original_shape = result.shape
            result_2d = result.reshape(-1, original_shape[-1])
            result_inv = self.scaler.inverse_transform(result_2d)
            result = result_inv.reshape(original_shape)

        # 2. 摄氏度转开尔文
        if self.kelvin_to_celsius:
            result = result + 273.15

        return result

    def fit_transform(self, train_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """拟合并转换训练数据"""
        self.fit(train_data)
        return self.transform(train_data)

    def save(self, path: str):
        """保存预处理器"""
        with open(path, 'wb') as f:
            pickle.dump({
                'kelvin_to_celsius': self.kelvin_to_celsius,
                'normalize': self.normalize,
                'scaler_type': self.scaler_type,
                'context_dim': self.context_dim,
                'scaler': self.scaler,
                'context_scaler': self.context_scaler,
                'fitted': self.fitted
            }, f)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        """加载预处理器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        preprocessor = DataPreprocessor(
            kelvin_to_celsius=data['kelvin_to_celsius'],
            normalize=data['normalize'],
            scaler_type=data['scaler_type'],
            context_dim=data.get('context_dim', 0)
        )
        preprocessor.scaler = data['scaler']
        preprocessor.context_scaler = data.get('context_scaler', None)
        preprocessor.fitted = data['fitted']

        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
