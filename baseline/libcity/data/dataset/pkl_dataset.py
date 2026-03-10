"""
PKL气象数据集加载器
加载HyperGKAN格式的.pkl数据集，实现与HyperGKAN完全一致的预处理流程。
支持StandardScaler标准化、开尔文到摄氏度转换、站点采样、样本采样。
"""
import os
import pickle
import numpy as np
from logging import getLogger
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir
from libcity.utils.normalization import Scaler


class SklearnBasedScaler(Scaler):
    """
    基于sklearn StandardScaler参数的包装器，与HyperGKAN的预处理完全对齐。
    使用sklearn fit得到的mean/std进行缩放，但在反向传播时保持计算图连续。

    关键：inverse_transform必须对torch tensor保持autograd兼容，
    否则训练时梯度无法反向传播！
    """
    def __init__(self, sklearn_scaler):
        self.sklearn_scaler = sklearn_scaler
        # 提取mean和std用于tensor运算
        self.mean = sklearn_scaler.mean_.astype(np.float32)
        self.std = sklearn_scaler.scale_.astype(np.float32)

    def transform(self, data):
        """仅在numpy数据预处理阶段使用，使用sklearn直接transform"""
        if isinstance(data, np.ndarray):
            original_shape = data.shape
            data_2d = data.reshape(-1, original_shape[-1])
            data_scaled = self.sklearn_scaler.transform(data_2d)
            return data_scaled.reshape(original_shape)
        else:
            # torch tensor - 使用算术运算保持autograd
            import torch
            mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.tensor(self.std, dtype=data.dtype, device=data.device)
            return (data - mean) / std

    def inverse_transform(self, data):
        """
        反标准化：data * std + mean
        必须对torch tensor保持autograd兼容（模型训练时loss计算需要梯度）
        """
        if isinstance(data, np.ndarray):
            original_shape = data.shape
            data_2d = data.reshape(-1, original_shape[-1])
            data_inv = self.sklearn_scaler.inverse_transform(data_2d)
            return data_inv.reshape(original_shape)
        else:
            # torch tensor - 使用算术运算保持autograd计算图
            import torch
            mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.tensor(self.std, dtype=data.dtype, device=data.device)
            return data * std + mean


# ==================== 气象要素配置 ====================
ELEMENT_SETTINGS = {
    "Temperature": {
        "data_dir": "temperature",
        "output_dim": 1,
        "kelvin_to_celsius": True,
    },
    "Cloud": {
        "data_dir": "cloud_cover",
        "output_dim": 1,
        "kelvin_to_celsius": False,
    },
    "Humidity": {
        "data_dir": "humidity",
        "output_dim": 1,
        "kelvin_to_celsius": False,
    },
    "Wind": {
        "data_dir": "component_of_wind",
        "output_dim": 2,
        "kelvin_to_celsius": False,
    },
}

# 数据根目录（使用相对路径，支持服务器结构）
# 服务器目录结构: hyper_kan/, baseline/, datasets/
# 当前在 baseline/ 目录下，数据在 ../datasets/ 下
DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../../../datasets")


def _haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点之间的球面距离(km)"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


class PklDataset(AbstractDataset):
    """
    加载PKL格式气象数据的数据集类。
    与HyperGKAN使用完全相同的数据和预处理流程。
    """

    def __init__(self, config):
        self.config = config
        self._logger = getLogger()

        # 基本参数
        self.batch_size = self.config.get('batch_size', 16)
        self.num_workers = self.config.get('num_workers', 0)
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)

        # 气象要素设置
        self.element = self.config.get('element', 'Temperature')
        if self.element not in ELEMENT_SETTINGS:
            raise ValueError(f"Unknown element: {self.element}. "
                             f"Valid: {list(ELEMENT_SETTINGS.keys())}")
        self.element_cfg = ELEMENT_SETTINGS[self.element]
        self.output_dim = self.element_cfg['output_dim']
        self.kelvin_to_celsius = self.element_cfg['kelvin_to_celsius']

        # 数据路径
        data_dir = os.path.join(DATA_ROOT, self.element_cfg['data_dir'])
        self.train_path = os.path.join(data_dir, 'trn.pkl')
        self.val_path = os.path.join(data_dir, 'val.pkl')
        self.test_path = os.path.join(data_dir, 'test.pkl')
        self.position_path = os.path.join(data_dir, 'position.pkl')

        # Context特征配置
        context_cfg = self.config.get('context_features', {})
        self.context_mask = [
            context_cfg.get('use_longitude', True),
            context_cfg.get('use_latitude', True),
            context_cfg.get('use_altitude', True),
            context_cfg.get('use_year', True),
            context_cfg.get('use_month', True),
            context_cfg.get('use_day', True),
            context_cfg.get('use_hour', True),
            context_cfg.get('use_region', False),
        ]
        self.use_context = any(self.context_mask)

        # 采样配置
        self.num_stations = self.config.get('num_stations', None)
        self.train_sample_ratio = self.config.get('train_sample_ratio', 1.0)
        self.val_sample_ratio = self.config.get('val_sample_ratio', 1.0)
        self.test_sample_ratio = self.config.get('test_sample_ratio', 1.0)

        # 初始化
        self.scaler = None
        self.adj_mx = None
        self.num_nodes = 0
        self.feature_dim = 0
        self.ext_dim = 0
        self.num_batches = 0
        self.data = None

    def _load_pkl(self, path):
        """加载pkl文件"""
        self._logger.info(f"Loading {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _extract_xy_context(self, data):
        """
        从pkl数据中提取x, y, context
        PKL格式: dict with 'x', 'y', 'context' keys
        x/y shape: (num_samples, 12, num_stations, feature_dim)
        context shape: (num_samples, 12, num_stations, context_dim)
        """
        if isinstance(data, dict):
            x = np.array(data.get('x'))
            y = np.array(data.get('y'))
            context = np.array(data.get('context')) if 'context' in data else None
        else:
            raise ValueError(f"Unexpected pkl data type: {type(data)}")
        return x, y, context

    def _apply_kelvin_to_celsius(self, *arrays):
        """开尔文转摄氏度"""
        results = []
        for arr in arrays:
            if arr is not None:
                results.append(arr - 273.15)
            else:
                results.append(None)
        return results

    def _apply_context_mask(self, context):
        """应用context特征掩码"""
        if context is None:
            return None
        mask_len = min(len(self.context_mask), context.shape[-1])
        selected = [i for i in range(mask_len) if self.context_mask[i]]
        if len(selected) == 0:
            return None
        return context[..., selected]

    def _sample_data(self, x, y, ctx, ratio, split_name):
        """按比例随机采样样本（x, y, context使用相同的索引）"""
        if ratio >= 1.0:
            self._logger.info(f"  {split_name}: 使用全部样本 (ratio={ratio})")
            return x, y, ctx
        n = x.shape[0]
        n_sample = max(1, int(n * ratio))
        indices = np.random.choice(n, n_sample, replace=False)
        indices.sort()
        self._logger.info(f"  {split_name}: {n} -> {n_sample} samples ({ratio * 100:.0f}%)")
        ctx_sampled = ctx[indices] if ctx is not None else None
        return x[indices], y[indices], ctx_sampled

    def _get_station_indices(self, total_stations):
        """获取站点采样索引（全局一次性生成，确保train/val/test一致）"""
        if not hasattr(self, '_station_indices'):
            if self.num_stations is None or self.num_stations >= total_stations:
                self._station_indices = None
            else:
                self._station_indices = np.sort(
                    np.random.choice(total_stations, self.num_stations, replace=False))
                self._logger.info(f"[Sampling] Station sampling: {total_stations} -> {self.num_stations}")
                self._logger.info(f"  站点索引范围: {self._station_indices[0]} ~ {self._station_indices[-1]}")
        return self._station_indices

    def _apply_station_sampling(self, *arrays, position=None, total_stations=None):
        """对arrays和position应用站点采样"""
        if total_stations is None:
            total_stations = arrays[0].shape[2]
        indices = self._get_station_indices(total_stations)
        if indices is None:
            return arrays + (position,)

        results = []
        for arr in arrays:
            if arr is not None and arr.ndim >= 3:
                results.append(arr[:, :, indices, ...])
            else:
                results.append(arr)

        if position is not None:
            position = position[indices]

        return tuple(results) + (position,)

    def _build_adj_from_position(self, position):
        """从站点位置构建距离邻接矩阵"""
        n = position.shape[0]
        self._logger.info(f"Building adjacency matrix from position data ({n} stations)...")

        # 计算距离矩阵
        dist_mx = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = _haversine_distance(position[i, 0], position[i, 1],
                                        position[j, 0], position[j, 1])
                dist_mx[i, j] = d
                dist_mx[j, i] = d

        # 应用高斯核
        distances = dist_mx[dist_mx > 0].flatten()
        if len(distances) > 0:
            std = distances.std()
            if std > 0:
                adj_mx = np.exp(-np.square(dist_mx / std))
                adj_mx[adj_mx < 0.1] = 0  # 阈值过滤
            else:
                adj_mx = np.eye(n, dtype=np.float32)
        else:
            adj_mx = np.eye(n, dtype=np.float32)

        self._logger.info(f"Adjacency matrix built: shape={adj_mx.shape}, "
                          f"non-zero={np.count_nonzero(adj_mx)}/{n * n}")
        return adj_mx

    def get_data(self):
        """
        加载并预处理数据，返回DataLoader
        """
        self._logger.info("=" * 60)
        self._logger.info(f"Loading PKL dataset: {self.element}")
        self._logger.info(f"  Kelvin to Celsius: {self.kelvin_to_celsius}")
        self._logger.info(f"  Context mask: {self.context_mask}")
        selected_count = sum(self.context_mask)
        self._logger.info(f"  Selected context features: {selected_count}/8")
        self._logger.info("=" * 60)

        # 1. 加载数据
        trn_data = self._load_pkl(self.train_path)
        val_data = self._load_pkl(self.val_path)
        test_data = self._load_pkl(self.test_path)

        x_train, y_train, ctx_train = self._extract_xy_context(trn_data)
        x_val, y_val, ctx_val = self._extract_xy_context(val_data)
        x_test, y_test, ctx_test = self._extract_xy_context(test_data)

        self._logger.info(f"  Train: x={x_train.shape}, y={y_train.shape}")
        self._logger.info(f"  Val:   x={x_val.shape}, y={y_val.shape}")
        self._logger.info(f"  Test:  x={x_test.shape}, y={y_test.shape}")

        # 2. 开尔文转摄氏度（仅Temperature）
        if self.kelvin_to_celsius:
            self._logger.info(f"Converting Kelvin to Celsius...")
            self._logger.info(f"  Before: x_train range [{x_train.min():.2f}, {x_train.max():.2f}]")
            x_train, y_train = self._apply_kelvin_to_celsius(x_train, y_train)
            x_val, y_val = self._apply_kelvin_to_celsius(x_val, y_val)
            x_test, y_test = self._apply_kelvin_to_celsius(x_test, y_test)
            self._logger.info(f"  After:  x_train range [{x_train.min():.2f}, {x_train.max():.2f}]")

        # 3. 样本采样（x, y, context使用完全相同的随机索引）
        self._logger.info("=" * 60)
        self._logger.info("[Sampling] 样本采样...")
        x_train, y_train, ctx_train = self._sample_data(
            x_train, y_train, ctx_train, self.train_sample_ratio, "Train")
        x_val, y_val, ctx_val = self._sample_data(
            x_val, y_val, ctx_val, self.val_sample_ratio, "Val")
        x_test, y_test, ctx_test = self._sample_data(
            x_test, y_test, ctx_test, self.test_sample_ratio, "Test")
        self._logger.info("[OK] 样本采样完成")

        # 4. 站点采样（train/val/test使用完全相同的站点索引）
        self._logger.info("=" * 60)
        position = None
        if os.path.exists(self.position_path):
            pos_data = self._load_pkl(self.position_path)
            if isinstance(pos_data, dict):
                position = np.array(pos_data.get('lonlat', pos_data.get('position', None)))
            else:
                position = np.array(pos_data)

        total_stations = x_train.shape[2]
        x_train, y_train, ctx_train, position = self._apply_station_sampling(
            x_train, y_train, ctx_train, position=position, total_stations=total_stations)
        x_val, y_val, ctx_val, _ = self._apply_station_sampling(
            x_val, y_val, ctx_val, position=None, total_stations=total_stations)
        x_test, y_test, ctx_test, _ = self._apply_station_sampling(
            x_test, y_test, ctx_test, position=None, total_stations=total_stations)

        self.num_nodes = x_train.shape[2]
        self._logger.info(f"  Stations: {self.num_nodes}")
        self._logger.info(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

        # 5. 应用context特征掩码
        if self.use_context:
            ctx_train = self._apply_context_mask(ctx_train)
            ctx_val = self._apply_context_mask(ctx_val)
            ctx_test = self._apply_context_mask(ctx_test)
            if ctx_train is not None:
                ctx_dim = ctx_train.shape[-1]
                self._logger.info(f"  Context features: {ctx_dim} dimensions")
            else:
                self._logger.info("  No context features available")

        # 6. StandardScaler标准化（与HyperGKAN对齐：使用sklearn StandardScaler）
        self._logger.info("=" * 60)
        self._logger.info("Fitting StandardScaler (sklearn, same as HyperGKAN)...")
        weather_dim = self.output_dim

        # 对气象特征拟合scaler
        sklearn_scaler = SklearnStandardScaler()
        x_train_2d = x_train[..., :weather_dim].reshape(-1, weather_dim)
        sklearn_scaler.fit(x_train_2d)
        self._logger.info(f"  Weather scaler mean: {sklearn_scaler.mean_}")
        self._logger.info(f"  Weather scaler std:  {sklearn_scaler.scale_}")

        self.scaler = SklearnBasedScaler(sklearn_scaler)

        # 标准化气象特征
        def scale_weather(arr):
            original_shape = arr.shape
            arr_2d = arr.reshape(-1, weather_dim)
            arr_scaled = sklearn_scaler.transform(arr_2d)
            return arr_scaled.reshape(original_shape).astype(np.float32)

        x_train_w = scale_weather(x_train[..., :weather_dim])
        y_train_w = scale_weather(y_train[..., :weather_dim])
        x_val_w = scale_weather(x_val[..., :weather_dim])
        y_val_w = scale_weather(y_val[..., :weather_dim])
        x_test_w = scale_weather(x_test[..., :weather_dim])
        y_test_w = scale_weather(y_test[..., :weather_dim])

        # 7. 拼接context特征到x和y（如果有）
        #    注意：y也需要拼接context，因为部分模型（如STGCN）的多步预测
        #    需要从y中取出ext特征来构造下一步输入
        if self.use_context and ctx_train is not None:
            # 对context也做标准化
            ctx_scaler = SklearnStandardScaler()
            ctx_dim = ctx_train.shape[-1]
            ctx_2d = ctx_train.reshape(-1, ctx_dim)
            ctx_scaler.fit(ctx_2d)

            def scale_ctx(c):
                if c is None:
                    return None
                s = c.shape
                return ctx_scaler.transform(c.reshape(-1, ctx_dim)).reshape(s).astype(np.float32)

            ctx_train = scale_ctx(ctx_train)
            ctx_val = scale_ctx(ctx_val)
            ctx_test = scale_ctx(ctx_test)

            x_train_final = np.concatenate([x_train_w, ctx_train], axis=-1)
            x_val_final = np.concatenate([x_val_w, ctx_val], axis=-1)
            x_test_final = np.concatenate([x_test_w, ctx_test], axis=-1)

            # y也拼接context（用于多步预测时构造下一步输入）
            y_train_w = np.concatenate([y_train_w, ctx_train], axis=-1)
            y_val_w = np.concatenate([y_val_w, ctx_val], axis=-1)
            y_test_w = np.concatenate([y_test_w, ctx_test], axis=-1)

            self.ext_dim = ctx_dim
            self._logger.info(f"  Concatenated: weather({weather_dim}) + context({ctx_dim}) = {x_train_final.shape[-1]}")
        else:
            x_train_final = x_train_w
            x_val_final = x_val_w
            x_test_final = x_test_w
            self.ext_dim = 0

        self.feature_dim = x_train_final.shape[-1]

        # 8. 构建邻接矩阵
        self._logger.info("=" * 60)
        if position is not None and len(position) > 0:
            self.adj_mx = self._build_adj_from_position(position)
        else:
            self._logger.info("No position data, using identity adjacency matrix")
            self.adj_mx = np.eye(self.num_nodes, dtype=np.float32)

        # 9. 构建DataLoader
        self._logger.info("=" * 60)
        self._logger.info("Creating DataLoaders...")
        feature_name = {'X': 'float', 'y': 'float'}
        train_data = list(zip(x_train_final, y_train_w))
        eval_data = list(zip(x_val_final, y_val_w))
        test_data = list(zip(x_test_final, y_test_w))

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, feature_name,
                                self.batch_size, self.num_workers,
                                pad_with_last_sample=False)
        self.num_batches = len(self.train_dataloader)

        self._logger.info(f"  Train batches: {len(self.train_dataloader)}")
        self._logger.info(f"  Val batches:   {len(self.eval_dataloader)}")
        self._logger.info(f"  Test batches:  {len(self.test_dataloader)}")
        self._logger.info(f"  Feature dim:   {self.feature_dim} "
                          f"(weather={weather_dim} + context={self.ext_dim})")
        self._logger.info(f"  Output dim:    {self.output_dim}")
        self._logger.info(f"  Num nodes:     {self.num_nodes}")
        self._logger.info("=" * 60)

        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """返回数据集特征，与原始框架兼容"""
        return {
            "scaler": self.scaler,
            "adj_mx": self.adj_mx,
            "ext_dim": self.ext_dim,
            "num_nodes": self.num_nodes,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "num_batches": self.num_batches,
        }
