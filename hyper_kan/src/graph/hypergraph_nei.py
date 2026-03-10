"""
邻域超图构建 (Neighbourhood Hypergraph)
基于地理距离的KNN超图
"""
import numpy as np
import torch
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
import logging

# ANSI颜色代码
YELLOW = '\033[93m'
RESET = '\033[0m'

logger = logging.getLogger(__name__)


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, 
                       lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    计算球面距离 (Haversine公式)
    
    Args:
        lat1, lon1: 起点经纬度 (度)
        lat2, lon2: 终点经纬度 (度)
    
    Returns:
        距离 (km)
    """
    R = 6371.0  # 地球半径 (km)
    
    # 转换为弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    
    return distance


def compute_geodesic_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    计算站点之间的球面距离矩阵
    
    Args:
        positions: 站点位置 (N, 2) - [latitude, longitude]
    
    Returns:
        距离矩阵 (N, N)
    """
    N = positions.shape[0]
    dist_matrix = np.zeros((N, N))
    
    for i in range(N):
        lat1, lon1 = positions[i]
        for j in range(i + 1, N):
            lat2, lon2 = positions[j]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def build_neighbourhood_hypergraph(
    positions: np.ndarray,
    top_k: int = 5,
    method: str = "knn",
    use_geodesic: bool = True,
    weight_decay: float = 0.1,
    cache_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建邻域超图
    
    基于地理距离的KNN构建超边，每个站点作为中心节点，
    连接其K个最近邻居形成一个超边。
    
    Args:
        positions: 站点位置 (N, 2) - [latitude, longitude]
        top_k: 每个超边包含的节点数量 (包括中心节点)
        method: 构建方法 ("knn" 或 "radius")
        use_geodesic: 是否使用球面距离
        weight_decay: 超边权重衰减因子
        cache_path: 缓存路径 (可选)
    
    Returns:
        H: 关联矩阵 (N, E) - H[i,j]=1表示节点i在超边j中
        W: 超边权重 (E,)
    """
    # 尝试加载缓存
    if cache_path is not None:
        try:
            data = np.load(cache_path)
            H = torch.from_numpy(data['H']).float()
            W = torch.from_numpy(data['W']).float()
            logger.info(f"{YELLOW}✓ 沿用缓存的邻域超图: {cache_path}{RESET}")
            return H, W
        except:
            pass
    
    N = positions.shape[0]
    logger.info(f"{YELLOW}⚡ 正在重新构建邻域超图 (N={N}, K={top_k}, method={method}){RESET}")

    # 计算距离矩阵
    if use_geodesic:
        dist_matrix = compute_geodesic_distance_matrix(positions)
        logger.info("使用球面距离 (Haversine) 计算")
    else:
        # 欧氏距离
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(positions, positions, metric='euclidean')
        logger.info("使用欧氏距离计算")
    
    # 使用KNN构建超图
    if method == "knn":
        # 每个节点作为中心，找到K个最近邻 (包括自己)
        # 注意：需要找K+1个邻居，然后排除自己，保留K个
        nbrs = NearestNeighbors(n_neighbors=top_k, metric='precomputed')
        nbrs.fit(dist_matrix)
        
        distances, indices = nbrs.kneighbors(dist_matrix)
        
        # 构建关联矩阵
        E = N  # 每个节点对应一个超边
        H = np.zeros((N, E), dtype=np.float32)
        W = np.zeros(E, dtype=np.float32)
        
        for i in range(N):
            # 超边包含节点i及其K-1个最近邻
            neighbors = indices[i]  # (K,)
            H[neighbors, i] = 1.0
            
            # 超边权重：基于距离的加权平均
            neighbor_distances = distances[i]
            # 使用指数衰减
            weights = np.exp(-weight_decay * neighbor_distances)
            W[i] = weights.mean()
        
        logger.info(f"KNN hypergraph built: {E} hyperedges")
    
    elif method == "radius":
        raise NotImplementedError("Radius-based hypergraph construction not implemented yet")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 转换为torch张量
    H = torch.from_numpy(H).float()
    W = torch.from_numpy(W).float()
    
    # 保存缓存
    if cache_path is not None:
        np.savez(cache_path, H=H.numpy(), W=W.numpy())
        logger.info(f"{YELLOW}💾 邻域超图已缓存至: {cache_path}{RESET}")

    # 统计信息
    avg_degree = (H.sum(dim=0).mean()).item()
    logger.info(f"平均超边大小: {avg_degree:.2f}")
    
    return H, W
