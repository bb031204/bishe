"""
语义超图构建 (Semantic Hypergraph)
基于时间序列特征相似度的超图
"""
import numpy as np
import torch
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import os
import logging

# ANSI颜色代码
YELLOW = '\033[93m'
RESET = '\033[0m'

logger = logging.getLogger(__name__)


def compute_feature_similarity(
    features: np.ndarray,
    similarity: str = "euclidean",
    normalize: bool = True
) -> np.ndarray:
    """
    计算特征相似度矩阵
    
    Args:
        features: 特征矩阵 (N, F) - N个站点，F维特征
        similarity: 相似度度量 ("euclidean", "pearson", "cosine")
        normalize: 是否标准化特征
    
    Returns:
        相似度矩阵 (N, N) - 值越大表示越相似
    """
    N = features.shape[0]
    
    # 标准化特征
    if normalize:
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True) + 1e-8
        features = (features - mean) / std
    
    if similarity == "euclidean":
        # 欧氏距离 -> 相似度 (距离越小越相似)
        dist_matrix = cdist(features, features, metric='euclidean')
        # 转换为相似度 (使用负距离的指数)
        sim_matrix = np.exp(-dist_matrix / dist_matrix.std())
    
    elif similarity == "pearson":
        # Pearson相关系数
        from scipy.stats import pearsonr
        sim_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                corr, _ = pearsonr(features[i], features[j])
                sim_matrix[i, j] = corr
                sim_matrix[j, i] = corr
        # 将相关系数转换为[0, 1]范围
        sim_matrix = (sim_matrix + 1) / 2
    
    elif similarity == "cosine":
        # 余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(features)
        # 转换为[0, 1]范围
        sim_matrix = (sim_matrix + 1) / 2
    
    else:
        raise ValueError(f"Unknown similarity metric: {similarity}")
    
    return sim_matrix


def build_semantic_hypergraph(
    train_data: np.ndarray,
    top_k: int = 5,
    similarity: str = "euclidean",
    input_window: int = 12,
    normalize_features: bool = True,
    cache_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建语义超图
    
    基于历史时间序列的相似度构建超边。
    相似的站点（即使地理位置远）会被连接在同一个超边中。
    
    Args:
        train_data: 训练数据 (T, N, F) - T个时间步，N个站点，F维特征
        top_k: 每个超边包含的节点数量
        similarity: 相似度度量方法
        input_window: 用于计算相似度的历史窗口长度
        normalize_features: 是否标准化特征
        cache_path: 缓存路径 (可选)
    
    Returns:
        H: 关联矩阵 (N, E)
        W: 超边权重 (E,)
    """
    # 尝试加载缓存
    if cache_path is not None:
        try:
            data = np.load(cache_path)
            H = torch.from_numpy(data['H']).float()
            W = torch.from_numpy(data['W']).float()
            logger.info(f"{YELLOW}✓ 沿用缓存的语义超图: {cache_path}{RESET}")
            return H, W
        except:
            pass
    
    # 处理不同的数据格式
    if train_data.ndim == 4:
        # (样本数, 时间步, 站点数, 特征数) -> (时间步*样本数, 站点数, 特征数)
        logger.info(f"输入数据形状: {train_data.shape} (4D: 样本数, 时间步, 站点数, 特征数)")
        S, T, N, F = train_data.shape
        # 重塑为 (S*T, N, F)，然后转置为 (T*S, N, F)
        train_data = train_data.reshape(S * T, N, F)
        T = S * T  # 总时间步数
        logger.info(f"重塑为: ({T}, {N}, {F})")
    elif train_data.ndim == 3:
        # (时间步, 站点数, 特征数)
        T, N, F = train_data.shape
    else:
        raise ValueError(f"意外的train_data形状: {train_data.shape}, 期望 (T, N, F) 或 (S, T, N, F)")

    logger.info(f"{YELLOW}⚡ 正在重新构建语义超图 (N={N}, K={top_k}, similarity={similarity}){RESET}")
    
    # 提取特征：使用最近input_window个时间步的数据
    # 将时间序列展平为特征向量
    if T >= input_window:
        recent_data = train_data[-input_window:, :, :]  # (input_window, N, F)
    else:
        recent_data = train_data  # (T, N, F)
        logger.warning(f"Insufficient time steps: T={T} < input_window={input_window}")
    
    # 展平: (N, input_window * F)
    features = recent_data.transpose(1, 0, 2).reshape(N, -1)
    
    logger.info(f"Feature shape for similarity computation: {features.shape}")
    
    # 计算相似度矩阵
    sim_matrix = compute_feature_similarity(
        features,
        similarity=similarity,
        normalize=normalize_features
    )
    
    # 转换为距离矩阵 (用于KNN)
    # 距离 = 1 - 相似度
    dist_matrix = 1.0 - sim_matrix
    
    # 使用KNN构建超图
    nbrs = NearestNeighbors(n_neighbors=top_k, metric='precomputed')
    nbrs.fit(dist_matrix)
    
    distances, indices = nbrs.kneighbors(dist_matrix)
    
    # 构建关联矩阵
    E = N  # 每个节点对应一个超边
    H = np.zeros((N, E), dtype=np.float32)
    W = np.zeros(E, dtype=np.float32)
    
    for i in range(N):
        # 超边包含节点i及其K-1个最相似的节点
        neighbors = indices[i]  # (K,)
        H[neighbors, i] = 1.0
        
        # 超边权重：基于相似度
        neighbor_similarities = 1.0 - distances[i]  # 转换回相似度
        W[i] = neighbor_similarities.mean()
    
    logger.info(f"Semantic hypergraph built: {E} hyperedges")
    
    # 转换为torch张量
    H = torch.from_numpy(H).float()
    W = torch.from_numpy(W).float()
    
    # 保存缓存
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, H=H.numpy(), W=W.numpy())
        logger.info(f"{YELLOW}💾 语义超图已缓存至: {cache_path}{RESET}")

    # 统计信息
    avg_degree = (H.sum(dim=0).mean()).item()
    avg_weight = W.mean().item()
    logger.info(f"平均超边大小: {avg_degree:.2f}")
    logger.info(f"平均超边权重: {avg_weight:.4f}")
    
    return H, W
