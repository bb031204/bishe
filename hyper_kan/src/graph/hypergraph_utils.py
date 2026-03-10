"""
超图工具函数
"""
import torch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_hypergraph_degrees(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算超图的节点度和超边度
    
    Args:
        H: 关联矩阵 (N, E) - N个节点，E条超边
    
    Returns:
        D_v: 节点度矩阵 (N, N) - 对角矩阵
        D_e: 超边度矩阵 (E, E) - 对角矩阵
    """
    # 节点度: d(v) = sum_e H(v,e)
    d_v = H.sum(dim=1)  # (N,)
    
    # 超边度: delta(e) = sum_v H(v,e)
    d_e = H.sum(dim=0)  # (E,)
    
    # 转换为对角矩阵
    D_v = torch.diag(d_v)  # (N, N)
    D_e = torch.diag(d_e)  # (E, E)
    
    return D_v, D_e


def normalize_hypergraph(
    H: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    add_self_loop: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    归一化超图 (用于超图卷积)
    
    计算: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    
    Args:
        H: 关联矩阵 (N, E)
        W: 超边权重 (E,) 或 (E, E)，如果为None则默认为1
        add_self_loop: 是否添加自环
    
    Returns:
        H_norm: 归一化后的关联矩阵
        D_v: 节点度矩阵
        D_e: 超边度矩阵
    """
    N, E = H.shape
    device = H.device
    
    # 添加自环
    if add_self_loop:
        H_self = torch.cat([H, torch.eye(N, device=device)], dim=1)  # (N, E+N)
        if W is not None:
            if W.dim() == 1:
                W_self = torch.cat([W, torch.ones(N, device=device)])  # (E+N,)
            else:
                W_self = torch.block_diag(W, torch.eye(N, device=device))  # (E+N, E+N)
        else:
            W_self = None
        H = H_self
        W = W_self
        E = E + N
    
    # 处理权重
    if W is not None:
        if W.dim() == 1:
            # (E,) -> (E, E) 对角矩阵
            W_diag = torch.diag(W)
        else:
            W_diag = W  # 已经是 (E, E)
    else:
        W_diag = torch.eye(E, device=device)
    
    # 计算度矩阵
    # 加权节点度: d(v) = sum_e w(e) * H(v,e)
    d_v = H @ W_diag.diag()  # (N,)
    d_v = torch.clamp(d_v, min=1e-8)  # 避免除零
    
    # 加权超边度: delta(e) = sum_v H(v,e)
    d_e = H.sum(dim=0)  # (E,)
    d_e = torch.clamp(d_e, min=1e-8)
    
    # 度矩阵
    D_v = torch.diag(d_v)  # (N, N)
    D_e = torch.diag(d_e)  # (E, E)
    
    return H, D_v, D_e


def hypergraph_laplacian(
    H: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    normalized: bool = True
) -> torch.Tensor:
    """
    计算超图拉普拉斯矩阵
    
    L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    
    Args:
        H: 关联矩阵 (N, E)
        W: 超边权重 (E,)
        normalized: 是否归一化
    
    Returns:
        L: 拉普拉斯矩阵 (N, N)
    """
    N, E = H.shape
    device = H.device
    
    # 处理权重
    if W is not None:
        if W.dim() == 1:
            W_diag = torch.diag(W)
        else:
            W_diag = W
    else:
        W_diag = torch.eye(E, device=device)
    
    # 计算度矩阵
    d_v = H @ W_diag.diag()
    d_v = torch.clamp(d_v, min=1e-8)
    
    d_e = H.sum(dim=0)
    d_e = torch.clamp(d_e, min=1e-8)
    
    if normalized:
        # 归一化拉普拉斯
        D_v_inv_sqrt = torch.diag(d_v ** (-0.5))
        D_e_inv = torch.diag(d_e ** (-1.0))
        
        # L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        L = torch.eye(N, device=device) - D_v_inv_sqrt @ H @ W_diag @ D_e_inv @ H.T @ D_v_inv_sqrt
    else:
        # 非归一化拉普拉斯
        D_v_mat = torch.diag(d_v)
        D_e_inv = torch.diag(d_e ** (-1.0))
        
        # L = D_v - H W D_e^{-1} H^T
        L = D_v_mat - H @ W_diag @ D_e_inv @ H.T
    
    return L


def visualize_hypergraph(H: torch.Tensor, save_path: Optional[str] = None):
    """
    可视化超图结构 (统计信息)
    
    Args:
        H: 关联矩阵 (N, E)
        save_path: 保存路径
    """
    N, E = H.shape
    
    # 节点度分布
    node_degrees = H.sum(dim=1).cpu().numpy()
    
    # 超边大小分布
    hyperedge_sizes = H.sum(dim=0).cpu().numpy()
    
    logger.info(f"Hypergraph Statistics:")
    logger.info(f"  Nodes: {N}")
    logger.info(f"  Hyperedges: {E}")
    logger.info(f"  Avg node degree: {node_degrees.mean():.2f} ± {node_degrees.std():.2f}")
    logger.info(f"  Avg hyperedge size: {hyperedge_sizes.mean():.2f} ± {hyperedge_sizes.std():.2f}")
    
    if save_path is not None:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 节点度分布
        axes[0].hist(node_degrees, bins=20, edgecolor='black')
        axes[0].set_xlabel('Node Degree')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Node Degree Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # 超边大小分布
        axes[1].hist(hyperedge_sizes, bins=20, edgecolor='black')
        axes[1].set_xlabel('Hyperedge Size')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Hyperedge Size Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Hypergraph visualization saved to {save_path}")
