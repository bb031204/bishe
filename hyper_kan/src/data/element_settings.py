"""
气象要素自适应科学配置模块 (Adaptive Scientific Settings)

根据论文 "Hypergraph Kolmogorov-Arnold Networks for Station-Level Meteorological Forecasting"
(Physica A 674, 2025) 的实验结论和物理常识，为不同气象要素自动匹配预处理策略和超参数。

核心科学依据：
  - 论文 Section 4.4 (Figure 3)：不同要素对超边节点数 K 的敏感度不同
  - 湿度 (Humidity): K_nei=2 效果最佳（强局部差异性，过大K引入噪声）
  - 温度/云量/风速: K_nei=5（受大尺度大气系统影响，论文实验设定）
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ==================== 数据集根路径 ====================
# 使用相对路径，从 hyper_kan/ 目录访问上层目录的 datasets/ 文件夹
DATA_ROOT = "../datasets"

# ==================== 各要素科学配置 ====================
# 每个要素的配置字典包含：
#   - data_dir: 数据文件夹名称（相对于 DATA_ROOT）
#   - output_dim: 输出特征维度
#   - kelvin_to_celsius: 是否需要开尔文转摄氏度（物理单位转换）
#   - normalize: 是否标准化
#   - scaler_type: 标准化类型
#   - neighbourhood_top_k: 邻域超图的 K 值（论文 Section 4.4）
#   - semantic_top_k: 语义超图的 K 值
#   - description: 科学说明

ELEMENT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "Temperature": {
        "data_dir": "temperature",
        "output_dim": 1,
        # 物理预处理：温度数据以开尔文为单位，需转摄氏度
        "kelvin_to_celsius": True,
        "normalize": True,
        "scaler_type": "standard",
        # 超图 K 值（论文实验设定：温度 K=5）
        "neighbourhood_top_k": 5,
        "semantic_top_k": 5,
        # ---- 数值稳定性参数 ----
        # 温度语义超图连通性良好，标准精度即可
        "numerical": {
            "conv_float32_norm": True,   # 归一化矩阵使用 float32（AMP安全）
            "degree_clamp_min": 1e-6,    # 节点度下界（防止除零）
        },
        # 气象说明
        "unit": "K -> °C",
        "description": "温度受大尺度大气系统影响，K=5能最好地捕捉高阶空间依赖",
    },

    "Cloud": {
        "data_dir": "cloud_cover",
        "output_dim": 1,
        # 物理预处理：云量为比例值 [0, 1]，无需单位转换
        "kelvin_to_celsius": False,
        "normalize": True,
        "scaler_type": "standard",
        # 超图 K 值（论文实验设定：云量 K=5）
        "neighbourhood_top_k": 5,
        "semantic_top_k": 5,
        # ---- 数值稳定性参数 ----
        # 云量值域 [0,1] 窄，语义超图存在大量零度节点（~34/768），
        # 必须使用 float32 归一化 + 较高 clamp 下界，否则 AMP float16 下
        # 1e-8 下溢为 0 → d_v^(-0.5)=Inf → NaN
        "numerical": {
            "conv_float32_norm": True,   # 必须开启：零度节点 + float16 = NaN
            "degree_clamp_min": 1e-6,    # 安全下界（float16 最小正数 ~5.96e-8）
        },
        # 气象说明
        "unit": "fraction [0,1]",
        "description": "云量受大尺度天气系统影响，K=5能捕捉区域性云系演变",
    },

    "Humidity": {
        "data_dir": "humidity",
        "output_dim": 1,
        # 物理预处理：相对湿度(%)，无需单位转换，但需标准化
        "kelvin_to_celsius": False,
        "normalize": True,
        "scaler_type": "standard",
        # 超图 K 值（论文 Figure 3：湿度在 K=2 时效果最佳）
        # 科学依据：湿度具有强局部差异性和快速变化特征，过大的K会引入噪声
        "neighbourhood_top_k": 2,
        "semantic_top_k": 2,
        # ---- 数值稳定性参数 ----
        # 湿度局部差异大且 K=2，语义超图可能存在孤立节点，需要 float32 保护
        "numerical": {
            "conv_float32_norm": True,   # 开启：K=2 时孤立节点概率更高
            "degree_clamp_min": 1e-6,
        },
        # 气象说明
        "unit": "%",
        "description": "湿度具有强局部差异性，K=2避免过大超边引入噪声(论文 Section 4.4)",
    },

    "Wind": {
        "data_dir": "component_of_wind",
        "output_dim": 2,  # 风有 u, v 两个分量
        # 物理预处理：风速分量(m/s)，无需单位转换
        "kelvin_to_celsius": False,
        "normalize": True,
        "scaler_type": "standard",
        # 超图 K 值（论文实验设定：风速 K=5）
        "neighbourhood_top_k": 5,
        "semantic_top_k": 5,
        # ---- 数值稳定性参数 ----
        # 风速 u/v 分量可能出现大量零值，语义超图零度节点风险中等
        "numerical": {
            "conv_float32_norm": True,   # 开启：安全保护
            "degree_clamp_min": 1e-6,
        },
        # 气象说明
        "unit": "m/s (u, v components)",
        "description": "风速受大尺度大气环流影响，K=5能捕捉高阶空间依赖",
    },
}


def get_element_settings(element: str) -> Dict[str, Any]:
    """
    获取指定气象要素的科学配置

    Args:
        element: 气象要素名称 ('Temperature', 'Cloud', 'Humidity', 'Wind')

    Returns:
        配置字典

    Raises:
        ValueError: 如果要素名称无效
    """
    if element not in ELEMENT_SETTINGS:
        valid = list(ELEMENT_SETTINGS.keys())
        raise ValueError(
            f"Unknown element: '{element}'. "
            f"Valid elements: {valid}"
        )

    settings = ELEMENT_SETTINGS[element].copy()

    # 自动生成数据路径
    data_dir = os.path.join(DATA_ROOT, settings["data_dir"])
    settings["train_path"] = os.path.join(data_dir, "trn.pkl")
    settings["val_path"] = os.path.join(data_dir, "val.pkl")
    settings["test_path"] = os.path.join(data_dir, "test.pkl")
    settings["position_path"] = os.path.join(data_dir, "position.pkl")

    return settings


def resolve_active_element(config: dict) -> str:
    """
    从 config 的 dataset_selection 中解析当前激活的要素

    Args:
        config: 完整配置字典

    Returns:
        激活的要素名称

    Raises:
        ValueError: 如果没有恰好一个要素被激活
    """
    selection = config.get("dataset_selection", None)

    if selection is None:
        # 向后兼容：如果没有 dataset_selection 字段，使用 meta.element
        element = config.get("meta", {}).get("element", "Temperature")
        logger.info(f"No dataset_selection found, using meta.element: {element}")
        return element

    active_elements = [name for name, active in selection.items() if active]

    if len(active_elements) == 0:
        raise ValueError(
            "dataset_selection 中没有任何要素被激活 (全部为 false)！"
            "请设置恰好一个要素为 true。"
        )

    if len(active_elements) > 1:
        raise ValueError(
            f"dataset_selection 中有多个要素被激活: {active_elements}！"
            "每次训练/预测只能有一个要素为 true。"
        )

    return active_elements[0]


def apply_element_settings(config: dict, logger_inst=None) -> dict:
    """
    根据 dataset_selection 自动应用科学配置到 config 中

    该函数会：
    1. 解析激活的要素
    2. 自动设置数据路径
    3. 自动设置预处理参数
    4. 自动设置超图 K 值
    5. 自动设置输出维度
    6. 更新 meta.element 和 experiment_name

    Args:
        config: 原始配置字典（会被就地修改）
        logger_inst: 可选的 logger 实例

    Returns:
        修改后的 config 字典
    """
    log = logger_inst or logger

    # 1. 解析激活要素
    element = resolve_active_element(config)
    settings = get_element_settings(element)

    log.info("=" * 50)
    log.info(f"Applying element-specific settings for: {element}")
    log.info(f"  Description: {settings['description']}")
    log.info(f"  Unit: {settings['unit']}")
    log.info(f"  Output dim: {settings['output_dim']}")

    # 2. 更新 meta
    config["meta"]["element"] = element
    config["meta"]["experiment_name"] = f"HyperGKAN_{element}"

    # 3. 更新数据路径
    config["data"]["train_path"] = settings["train_path"]
    config["data"]["val_path"] = settings["val_path"]
    config["data"]["test_path"] = settings["test_path"]
    config["data"]["position_path"] = settings["position_path"]
    log.info(f"  Data dir: {os.path.dirname(settings['train_path'])}")

    # 4. 更新预处理参数
    config["data"]["kelvin_to_celsius"] = settings["kelvin_to_celsius"]
    config["data"]["normalize"] = settings["normalize"]
    config["data"]["scaler_type"] = settings["scaler_type"]
    log.info(f"  Kelvin to Celsius: {settings['kelvin_to_celsius']}")
    log.info(f"  Normalize: {settings['normalize']} ({settings['scaler_type']})")

    # 5. 更新超图 K 值
    config["graph"]["neighbourhood"]["top_k"] = settings["neighbourhood_top_k"]
    config["graph"]["semantic"]["top_k"] = settings["semantic_top_k"]
    log.info(f"  Neighbourhood K: {settings['neighbourhood_top_k']}")
    log.info(f"  Semantic K: {settings['semantic_top_k']}")

    # 6. 更新输出维度
    config["model"]["output_projection"]["output_dim"] = settings["output_dim"]
    log.info(f"  Output dim: {settings['output_dim']}")

    # 7. 更新超图卷积数值稳定性参数 (数据集特异性)
    numerical = settings.get("numerical", {})
    conv_cfg = config["graph"]["conv"]
    conv_cfg["float32_norm"] = numerical.get("conv_float32_norm", True)
    conv_cfg["degree_clamp_min"] = numerical.get("degree_clamp_min", 1e-6)
    log.info(f"  Conv float32 norm: {conv_cfg['float32_norm']}")
    log.info(f"  Degree clamp min: {conv_cfg['degree_clamp_min']}")

    log.info("=" * 50)

    return config


def validate_dataset_selection(config: dict) -> bool:
    """
    验证 dataset_selection 配置的正确性

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    selection = config.get("dataset_selection", None)
    if selection is None:
        return True  # 向后兼容

    valid_elements = set(ELEMENT_SETTINGS.keys())
    config_elements = set(selection.keys())

    # 检查是否有未知的要素名
    unknown = config_elements - valid_elements
    if unknown:
        logger.error(f"Unknown elements in dataset_selection: {unknown}")
        return False

    # 检查是否恰好有一个激活
    active = [k for k, v in selection.items() if v]
    if len(active) != 1:
        logger.error(
            f"dataset_selection must have exactly 1 active element, "
            f"found {len(active)}: {active}"
        )
        return False

    return True
