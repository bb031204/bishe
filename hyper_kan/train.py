"""
HyperGKAN 训练脚本
支持多卡 DataParallel 并行训练
"""
import os
import sys

# ====== GPU 环境配置（必须在 import torch 之前）======
# --gpu N      : 仅使用单张GPU N
# --gpus 0,1,2 : 使用指定的多张GPU (DataParallel)
# 默认(无参数)  : 使用所有可见GPU (DataParallel)
if '--gpu' in sys.argv:
    gpu_idx = sys.argv[sys.argv.index('--gpu') + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
elif '--gpus' in sys.argv:
    gpu_ids = sys.argv[sys.argv.index('--gpus') + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
# else: 不设置 CUDA_VISIBLE_DEVICES，使用所有可见GPU

import yaml
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import math
from datetime import datetime

# ANSI颜色代码
YELLOW = '\033[93m'
RESET = '\033[0m'

# 导入自定义模块
from src.data import load_pkl_data, load_position_data, create_data_loaders, DataPreprocessor
from src.data import apply_element_settings, validate_dataset_selection
from src.graph import build_neighbourhood_hypergraph, build_semantic_hypergraph
from src.models import HyperGKAN
from src.training import Trainer
from src.utils import setup_logger


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    # 如果是相对路径，相对于脚本所在目录
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict, logger):
    """准备数据"""
    logger.info("=" * 50)
    logger.info("Loading data...")
    
    # 构建context特征掩码
    context_feature_mask = None
    if config['data'].get('use_context', False) and 'context_features' in config['data']:
        context_features = config['data']['context_features']
        # 按顺序：经度、纬度、海拔、年、月、日、时、区域
        context_feature_mask = [
            context_features.get('use_longitude', True),
            context_features.get('use_latitude', True),
            context_features.get('use_altitude', True),
            context_features.get('use_year', True),
            context_features.get('use_month', True),
            context_features.get('use_day', True),
            context_features.get('use_hour', True),
            context_features.get('use_region', True)
        ]
        logger.info(f"Context feature mask: {context_feature_mask}")
        logger.info(f"  Selected features: {sum(context_feature_mask)}/8")
    
    # 加载数据
    train_data = load_pkl_data(
        config['data']['train_path'],
        use_context=config['data']['use_context'],
        context_dim=config['data']['context_dim'],
        use_dim4=config['data']['use_dim4'],
        context_feature_mask=context_feature_mask
    )
    
    val_data = load_pkl_data(
        config['data']['val_path'],
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
    
    # 加载位置信息
    position = load_position_data(config['data']['position_path'])
    
    logger.info(f"Data loaded successfully")
    logger.info(f"  Train: {train_data['x'].shape}")
    logger.info(f"  Val:   {val_data['x'].shape}")
    logger.info(f"  Test:  {test_data['x'].shape}")
    
    # 处理position数据
    if position is not None and hasattr(position, 'shape') and len(position.shape) > 0 and position.shape[0] > 0:
        num_stations = position.shape[0]
        logger.info(f"  Stations: {num_stations} (from position data)")
    else:
        # 从数据中推断站点数
        if train_data['x'].ndim == 4:
            # (样本数, 时间步, 站点数, 特征数)
            num_stations = train_data['x'].shape[2]
        elif train_data['x'].ndim == 3:
            # (时间步, 站点数, 特征数)
            num_stations = train_data['x'].shape[1]
        else:
            raise ValueError(f"Unexpected data shape: {train_data['x'].shape}")
        
        logger.warning(f"  Position data invalid or missing, inferring stations from data: {num_stations}")
        
        # 生成默认位置（均匀分布在网格上）
        grid_size = int(math.ceil(math.sqrt(num_stations)))
        positions = []
        for i in range(num_stations):
            lat = (i // grid_size) * (180.0 / grid_size) - 90.0  # -90 to 90
            lon = (i % grid_size) * (360.0 / grid_size) - 180.0  # -180 to 180
            positions.append([lat, lon])
        position = np.array(positions)
        logger.warning(f"  Generated default grid positions: {position.shape}")
    
    return train_data, val_data, test_data, position


def sample_samples(train_data: dict, val_data: dict, test_data: dict,
                   train_ratio: float = 1.0, val_ratio: float = 1.0, test_ratio: float = 1.0,
                   seed: int = 42, logger=None) -> tuple:
    """
    样本维度采样：从数据集中按比例随机选择样本

    Args:
        train_data: 训练数据字典
        val_data: 验证数据字典
        test_data: 测试数据字典
        train_ratio: 训练集采样比例 (0.0-1.0)
        val_ratio: 验证集采样比例 (0.0-1.0)
        test_ratio: 测试集采样比例 (0.0-1.0)
        seed: 随机种子
        logger: 日志记录器

    Returns:
        采样后的数据

    说明：
        - 在样本数量维度采样，与站点采样不同
        - 每个样本仍包含所有站点
        - 用于快速实验和调试
        - 例如: ratio=0.5 表示使用50%的样本
    """
    if logger is None:
        return train_data, val_data, test_data

    logger.info("=" * 50)
    logger.info("🔧 样本采样检查...")

    def sample_data_samples(data_dict: dict, sample_ratio: float, name: str) -> dict:
        """对单个数据集进行样本采样"""
        if sample_ratio >= 1.0:
            logger.info(f"  {name}: 使用全部样本 (ratio={sample_ratio})")
            return data_dict

        num_samples = data_dict['x'].shape[0]
        target_samples = int(num_samples * sample_ratio)
        target_samples = max(1, target_samples)  # 至少保留1个样本

        logger.info(f"  {name}: {num_samples} → {target_samples} 样本 ({sample_ratio*100:.1f}%)")

        # 设置随机种子
        rng = np.random.default_rng(seed)

        # 随机选择样本索引
        selected_indices = rng.choice(num_samples, target_samples, replace=False)
        selected_indices = np.sort(selected_indices)

        # 对数据进行采样
        sampled = {}
        for key, value in data_dict.items():
            if key in ['x', 'y']:
                sampled[key] = value[selected_indices]
                logger.info(f"    {key}: {value.shape} → {sampled[key].shape}")
            elif key == 'context':
                # context需要对应采样
                if value is not None and value.ndim > 0 and value.shape[0] == num_samples:
                    sampled[key] = value[selected_indices]
                    logger.info(f"    context: {value.shape} → {sampled[key].shape}")
                else:
                    sampled[key] = value
            else:
                sampled[key] = value

        return sampled

    # 对三份数据集分别采样
    train_data = sample_data_samples(train_data, train_ratio, "Train")
    val_data = sample_data_samples(val_data, val_ratio, "Val")
    test_data = sample_data_samples(test_data, test_ratio, "Test")

    logger.info("✓ 样本采样完成")

    return train_data, val_data, test_data


def sample_stations(train_data: dict, val_data: dict, test_data: dict,
                    position: np.ndarray, num_stations: int,
                    seed: int = 42, logger=None) -> tuple:
    """
    空间维度采样：从所有站点中随机选择指定数量的站点

    Args:
        train_data: 训练数据字典
        val_data: 验证数据字典
        test_data: 测试数据字典
        position: 站点位置 (N, 2)
        num_stations: 要保留的站点数量
        seed: 随机种子
        logger: 日志记录器

    Returns:
        采样后的数据和位置

    说明：
        - 只在空间维度采样，时间维度保持完整
        - 每个站点仍保留完整的12帧历史
        - 三份数据集使用相同的站点索引，确保一致性
    """
    if logger is None:
        return train_data, val_data, test_data, position

    # 获取原始站点数
    original_num_stations = position.shape[0]

    # 如果num_stations为null或大于等于原始站点数，不进行采样
    if num_stations is None or num_stations >= original_num_stations:
        logger.info(f"站点采样未启用，使用全部 {original_num_stations} 个站点")
        return train_data, val_data, test_data, position

    logger.info("=" * 50)
    logger.info(f"🔧 站点采样: {original_num_stations} → {num_stations}")

    # 设置随机种子确保可复现
    np.random.seed(seed)

    # 随机选择站点索引
    selected_indices = np.random.choice(
        original_num_stations,
        num_stations,
        replace=False
    )

    # 排序索引（可选，让站点顺序保持一致）
    selected_indices = np.sort(selected_indices)

    logger.info(f"  选择的站点索引范围: {selected_indices[0]} ~ {selected_indices[-1]}")

    # 对数据进行采样
    def sample_data(data_dict, name):
        """对单个数据集进行站点采样"""
        sampled = {}
        for key, value in data_dict.items():
            if key in ['x', 'y', 'context']:
                # 数据形状: (样本, 时间, 站点, 特征) 或 (时间, 站点, 特征)
                if value is not None:
                    if value.ndim == 4:
                        sampled[key] = value[:, :, selected_indices, :]
                    elif value.ndim == 3:
                        sampled[key] = value[:, selected_indices, :]
                    else:
                        sampled[key] = value
                    logger.info(f"  {name}.{key}: {value.shape} -> {sampled[key].shape}")
                else:
                    sampled[key] = None
            else:
                sampled[key] = value
        return sampled

    train_data = sample_data(train_data, "Train")
    val_data = sample_data(val_data, "Val")
    test_data = sample_data(test_data, "Test")

    # 对position进行采样
    position = position[selected_indices]
    logger.info(f"  Position: {(original_num_stations, 2)} → {position.shape}")

    logger.info(f"✓ 站点采样完成，时序特征(12帧)保持完整")

    return train_data, val_data, test_data, position


def get_hypergraph_cache_path(config: dict, graph_type: str, num_stations: int = None) -> str:
    """
    生成超图缓存路径（基于config参数 + 站点数）

    Args:
        config: 配置字典
        graph_type: 'neighbourhood' 或 'semantic'
        num_stations: 当前使用的站点数（采样后）

    Returns:
        缓存文件路径
    """
    cache_dir = config['graph']['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)

    element = config['meta']['element']

    if graph_type == 'neighbourhood':
        top_k = config['graph']['neighbourhood']['top_k']
        method = config['graph']['neighbourhood']['method']
        use_geodesic = config['graph']['neighbourhood']['use_geodesic']
        weight_decay = config['graph']['neighbourhood']['weight_decay']

        # 缓存名包含站点数，避免采样后错误复用
        station_suffix = f"_N{num_stations}" if num_stations else ""
        cache_name = f"{element}_nei_K{top_k}_{method}_geo{use_geodesic}_decay{weight_decay}{station_suffix}.npz"

    elif graph_type == 'semantic':
        top_k = config['graph']['semantic']['top_k']
        similarity = config['graph']['semantic']['similarity']
        input_window = config['graph']['semantic']['input_window']
        normalize = config['graph']['semantic']['normalize_features']

        # 缓存名包含站点数，避免采样后错误复用
        station_suffix = f"_N{num_stations}" if num_stations else ""
        cache_name = f"{element}_sem_K{top_k}_{similarity}_win{input_window}_norm{normalize}{station_suffix}.npz"

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    return os.path.join(cache_dir, cache_name)



def get_visualization_path(config: dict, graph_type: str) -> str:
    """
    获取可视化保存路径
    
    Args:
        config: 配置字典
        graph_type: 'neighbourhood' 或 'semantic'
    
    Returns:
        可视化文件路径
    """
    visual_dir = config['graph'].get('visual_dir', 'visuals')
    element = config['meta']['element']
    
    # 创建要素专用文件夹
    element_dir = os.path.join(visual_dir, element)
    os.makedirs(element_dir, exist_ok=True)
    
    if graph_type == 'neighbourhood':
        top_k = config['graph']['neighbourhood']['top_k']
        method = config['graph']['neighbourhood']['method']
        filename = f"hypergraph_neighbourhood_K{top_k}_{method}.png"
    else:  # semantic
        top_k = config['graph']['semantic']['top_k']
        similarity = config['graph']['semantic']['similarity']
        filename = f"hypergraph_semantic_K{top_k}_{similarity}.png"
    
    return os.path.join(element_dir, filename)


def build_hypergraphs(train_data: dict, position: np.ndarray, config: dict, logger, output_dir: str = None):
    """构建或加载超图"""
    logger.info("=" * 50)
    logger.info("超图构建/加载中...")

    # 获取当前使用的站点数
    num_stations = position.shape[0]
    logger.info(f"当前使用站点数: {num_stations}")

    # 邻域超图
    nei_cache_path = get_hypergraph_cache_path(config, 'neighbourhood', num_stations) if config['graph']['use_cache'] else None

    H_nei, W_nei = build_neighbourhood_hypergraph(
        positions=position,
        top_k=config['graph']['neighbourhood']['top_k'],
        method=config['graph']['neighbourhood']['method'],
        use_geodesic=config['graph']['neighbourhood']['use_geodesic'],
        weight_decay=config['graph']['neighbourhood']['weight_decay'],
        cache_path=nei_cache_path
    )

    logger.info(f"邻域超图形状: {H_nei.shape}")

    # 语义超图
    sem_cache_path = get_hypergraph_cache_path(config, 'semantic', num_stations) if config['graph']['use_cache'] else None

    H_sem, W_sem = build_semantic_hypergraph(
        train_data=train_data['x'],
        top_k=config['graph']['semantic']['top_k'],
        similarity=config['graph']['semantic']['similarity'],
        input_window=config['graph']['semantic']['input_window'],
        normalize_features=config['graph']['semantic']['normalize_features'],
        cache_path=sem_cache_path
    )

    logger.info(f"语义超图形状: {H_sem.shape}")
    
    # 可视化超图结构（如果启用）
    if config.get('graph', {}).get('visualize', False):
        from src.graph.hypergraph_utils import visualize_hypergraph

        logger.info("生成超图可视化...")

        # 获取可视化保存路径
        nei_viz_path = get_visualization_path(config, 'neighbourhood')
        sem_viz_path = get_visualization_path(config, 'semantic')

        # 检查是否已存在
        if os.path.exists(nei_viz_path):
            logger.info(f"✓ 邻域超图可视化已存在: {nei_viz_path}")
        else:
            visualize_hypergraph(H_nei, save_path=nei_viz_path)
            logger.info(f"✓ 邻域超图可视化已保存: {nei_viz_path}")

        if os.path.exists(sem_viz_path):
            logger.info(f"✓ 语义超图可视化已存在: {sem_viz_path}")
        else:
            visualize_hypergraph(H_sem, save_path=sem_viz_path)
            logger.info(f"✓ 语义超图可视化已保存: {sem_viz_path}")
    
    return H_nei, H_sem, W_nei, W_sem


def create_model(config: dict, input_dim: int, logger):
    """创建模型"""
    logger.info("=" * 50)
    logger.info("Creating model...")
    
    # 从 config 读取超图卷积数值稳定性参数 (由 element_settings.py 设置)
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
    
    num_params = model.get_num_parameters()
    logger.info(f"Model created: {num_params:,} trainable parameters")
    
    return model


def create_optimizer_and_scheduler(model, config: dict):
    """创建优化器和调度器"""
    # 优化器
    opt_config = config['training']['optimizer']
    
    if opt_config['type'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas']
        )
    elif opt_config['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas']
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    # 调度器
    sched_config = config['training']['scheduler']
    
    if sched_config['type'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config['mode'],
            factor=sched_config['factor'],
            patience=sched_config['patience'],
            min_lr=sched_config['min_lr']
        )
    elif sched_config['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=sched_config['min_lr']
        )
    elif sched_config['type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 50),
            gamma=sched_config['factor']
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def create_loss_function(config: dict):
    """创建损失函数"""
    loss_type = config['training']['loss']['type'].lower()
    
    if loss_type == 'mae' or loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse' or loss_type == 'l2':
        return nn.MSELoss()
    elif loss_type == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


def main(args):
    """主函数"""
    # 加载配置
    config = load_config(args.config)

    # 如果命令行指定了 GPU，覆盖 config 中的设置
    if args.gpu is not None:
        config['meta']['gpu'] = args.gpu

    # ====== 自适应科学配置：根据 dataset_selection 自动设置 ======
    # 验证 dataset_selection 互斥性
    if not validate_dataset_selection(config):
        print("ERROR: dataset_selection 配置无效，请确保恰好有一个要素为 true")
        return
    # 应用要素特定的科学配置（数据路径、预处理、超图K值、输出维度）
    config = apply_element_settings(config)

    # 恢复训练时，清除时间限制（避免继承旧的暂停设置）
    if args.resume:
        config['training']['time_limit_minutes'] = None

    # 设置随机种子
    if config['reproducibility']['deterministic']:
        set_seed(config['meta']['seed'])

    # 设置日志 - 使用脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 恢复训练时，使用原来的输出目录
    if args.resume:
        # 加载 checkpoint 获取原始输出目录
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        # 从 checkpoint 中获取原始 output_dir（如果有的话）
        # checkpoint 中保存的是完整路径，直接使用
        # 需要从 config 中推断原始目录结构
        saved_config = checkpoint.get('config', {})
        if saved_config:
            # 从保存的 config 中重建 output_dir
            # 由于我们无法直接获取原始时间戳，使用 checkpoint 所在目录
            checkpoint_dir = os.path.dirname(args.resume)
            output_dir = os.path.dirname(checkpoint_dir)
            # logger 还未设置，暂时不输出
        else:
            # fallback：创建新目录
            output_dir = os.path.join(
                script_dir,
                config['output']['base_dir'],
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['meta']['element']}"
            )
    else:
        # 新训练：创建新的输出目录
        output_dir = os.path.join(
            script_dir,
            config['output']['base_dir'],
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['meta']['element']}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # 复制config.yaml到输出目录（新训练时复制，恢复训练时保留原config）
    import shutil
    config_backup_path = os.path.join(output_dir, 'config.yaml')
    if not args.resume or not os.path.exists(config_backup_path):
        # 基于脚本目录解析config路径（而非当前工作目录）
        if os.path.isabs(args.config):
            config_abs_path = args.config
        else:
            config_abs_path = os.path.join(script_dir, args.config)
        shutil.copy2(config_abs_path, config_backup_path)

    logger = setup_logger(
        name="HyperGKAN",
        level=config['output']['logging']['level'],
        output_dir=output_dir if config['output']['logging']['file'] else None,
        console=config['output']['logging']['console'],
        file=config['output']['logging']['file'],
        append_mode=bool(args.resume)  # 恢复训练时使用追加模式
    )
    
    logger.info("=" * 50)
    logger.info("HyperGKAN Training")
    logger.info("=" * 50)
    logger.info(f"Experiment: {config['meta']['experiment_name']}")
    logger.info(f"Element: {config['meta']['element']}")
    logger.info(f"Device: {config['meta']['device']}")
    logger.info(f"Config saved to: {config_backup_path}")
    
    # 准备数据
    train_data, val_data, test_data, position = prepare_data(config, logger)

    # 样本采样（样本维度，按比例）
    train_sample_ratio = config['data'].get('train_sample_ratio', 1.0)
    val_sample_ratio = config['data'].get('val_sample_ratio', 1.0)
    test_sample_ratio = config['data'].get('test_sample_ratio', 1.0)

    if train_sample_ratio < 1.0 or val_sample_ratio < 1.0 or test_sample_ratio < 1.0:
        train_data, val_data, test_data = sample_samples(
            train_data, val_data, test_data,
            train_ratio=train_sample_ratio,
            val_ratio=val_sample_ratio,
            test_ratio=test_sample_ratio,
            seed=config['meta']['seed'],
            logger=logger
        )

    # 站点采样（空间维度）
    num_stations = config['data'].get('num_stations', None)
    if num_stations is not None:
        train_data, val_data, test_data, position = sample_stations(
            train_data, val_data, test_data, position,
            num_stations=num_stations,
            seed=config['meta']['seed'],
            logger=logger
        )

    # 数据预处理（温度转换 + 标准化）
    # 计算实际使用的context特征数量
    actual_context_dim = 0
    if config['data'].get('use_context', False) and train_data.get('context') is not None:
        actual_context_dim = train_data['context'].shape[-1]
        logger.info(f"Context features to be concatenated: {actual_context_dim} dimensions")
    
    preprocessor = DataPreprocessor(
        kelvin_to_celsius=config['data'].get('kelvin_to_celsius', False),
        normalize=config['data'].get('normalize', False),
        scaler_type=config['data'].get('scaler_type', 'standard'),
        context_dim=actual_context_dim
    )

    # 在训练数据上拟合
    train_data = preprocessor.fit_transform(train_data)

    # 转换验证和测试数据
    val_data = preprocessor.transform(val_data)
    test_data = preprocessor.transform(test_data)

    # 保存预处理器（用于预测时反标准化）
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    logger.info(f"Preprocessor saved to: {preprocessor_path}")

    # 构建超图
    H_nei, H_sem, W_nei, W_sem = build_hypergraphs(train_data, position, config, logger, output_dir)
    
    # 创建数据加载器
    logger.info("=" * 50)
    logger.info("Creating data loaders...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        input_window=config['data']['input_window'],
        output_window=config['data']['output_window'],
        batch_size=config['data']['batch_size'],
        num_workers=config['meta']['num_workers'],
        shuffle_train=config['data']['shuffle_train'],
        stride=1,
        concat_context=config['data'].get('use_context', False)  # 如果使用context则拼接
    )
    
    # 创建模型
    # 计算实际的输入维度：气象特征 + context特征
    weather_feat_dim = train_data['x'].shape[-1]
    if config['data'].get('use_context', False) and actual_context_dim > 0:
        input_dim = weather_feat_dim + actual_context_dim
    else:
        input_dim = weather_feat_dim
    
    logger.info(f"Model input dimension: {input_dim}")
    if actual_context_dim > 0:
        logger.info(f"  = Weather features ({weather_feat_dim}) + Context features ({actual_context_dim})")
    else:
        logger.info(f"  = Weather features only ({weather_feat_dim})")
    
    model = create_model(config, input_dim, logger)
    
    # ====== 多卡 DataParallel 并行 ======
    # 注意: DataParallel 仅在显式使用 --multi_gpu 时启用
    # 对于小模型(HyperGKAN) + 小batch(16)，推荐单卡训练 + 多卡并行跑不同数据集
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = getattr(args, 'multi_gpu', False) and (num_gpus > 1)
    
    if use_multi_gpu:
        logger.info("=" * 50)
        logger.info(f"🚀 Multi-GPU DataParallel: 检测到 {num_gpus} 张GPU")
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        for i, name in enumerate(gpu_names):
            logger.info(f"  GPU {i}: {name}")
        
        # ★ 关键: 将超图矩阵注册为模型buffer，避免DataParallel错误切分
        # H_nei (N, E) 不含batch维度，如果作为forward参数传入会被沿dim=0切分节点维！
        device = config['meta']['device']
        model.set_hypergraph_buffers(
            H_nei.to(device), H_sem.to(device),
            W_nei.to(device) if W_nei is not None else None,
            W_sem.to(device) if W_sem is not None else None
        )
        model = nn.DataParallel(model)
        logger.info(f"模型已包装为 DataParallel，batch_size={config['data']['batch_size']} "
                     f"将自动分配到 {num_gpus} 张GPU")
    elif torch.cuda.is_available():
        logger.info(f"🖥️  Single-GPU 模式: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️  CUDA 不可用，使用 CPU")
    
    # 创建优化器和调度器（注意：DataParallel模型的parameters()仍可正常使用）
    logger.info("=" * 50)
    logger.info("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    logger.info(f"Optimizer: {config['training']['optimizer']['type']}")
    logger.info(f"Scheduler: {config['training']['scheduler']['type']}")
    logger.info(f"Initial LR: {config['training']['optimizer']['lr']}")
    
    # 创建损失函数
    loss_fn = create_loss_function(config)
    logger.info(f"Loss function: {config['training']['loss']['type']}")
    
    # 创建训练器
    logger.info("=" * 50)
    logger.info("Creating trainer...")
    
    device = config['meta']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        H_nei=H_nei,
        H_sem=H_sem,
        W_nei=W_nei,
        W_sem=W_sem,
        device=device,
        config=config,
        preprocessor=preprocessor,
        output_dir=output_dir  # 传入输出目录（恢复训练时使用原目录）
    )
    
    # 开始训练
    trainer.train(resume_from=args.resume)
    
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Output directory: {trainer.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperGKAN Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='单GPU模式: 指定使用哪张GPU (overrides config)'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default=None,
        help='多GPU模式: 指定使用的GPU编号，逗号分隔 (如 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        default=False,
        help='启用 DataParallel 多卡并行 (需显式指定，默认关闭)'
    )

    args = parser.parse_args()
    main(args)
