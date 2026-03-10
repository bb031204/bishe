"""
KAN (Kolmogorov-Arnold Networks) 层实现
基于pykan库的封装
"""
import torch
import torch.nn as nn
from typing import List, Optional
import logging

# ANSI颜色代码
YELLOW = '\033[93m'
RESET = '\033[0m'

logger = logging.getLogger(__name__)


class KANLinear(nn.Module):
    """
    KAN线性层 (替代传统MLP)
    
    使用可学习的单变量样条函数替代固定权重
    增强版：支持自动chunking和fallback机制
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 noise_scale: float = 0.1,
                 base_activation: str = "silu",
                 chunk_size: int = 512):  # 新增：chunk大小
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            grid_size: B-spline网格大小
            spline_order: 样条阶数
            noise_scale: 噪声尺度
            base_activation: 基础激活函数
            chunk_size: 处理大批量数据时的分块大小
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.chunk_size = chunk_size
        self.oom_count = 0  # 统计OOM次数
        
        # 预先创建fallback层，避免OOM时再创建
        self.linear_fallback = nn.Linear(in_features, out_features)
        
        try:
            from kan import KAN
            
            # 创建KAN模型
            # width: [in_features, out_features]
            self.kan = KAN(
                width=[in_features, out_features],
                grid=grid_size,
                k=spline_order,
                noise_scale=noise_scale,
                base_fun=base_activation,
                device='cpu'  # 将在forward时移动到正确设备
            )
            
            self.use_kan = True
            logger.debug(f"KANLinear initialized: {in_features} -> {out_features}, chunk_size={chunk_size}")
        
        except ImportError:
            logger.warning(f"{YELLOW}⚠️ pykan未安装，自动切换为标准Linear层{RESET}")
            self.use_kan = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (..., in_features)
        
        Returns:
            输出张量 (..., out_features)
        """
        if not self.use_kan:
            # 直接使用线性层
            return self.linear_fallback(x)
        
        # 如果OOM次数过多，永久切换到线性层
        if self.oom_count >= 3:
            if self.oom_count == 3:
                logger.warning(f"{YELLOW}⚠️ KAN层发生{self.oom_count}次显存不足错误，永久切换为Linear层{RESET}")
                self.oom_count += 1  # 避免重复打印
            return self.linear_fallback(x)
        
        # KAN期望输入为 (batch, in_features)
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        batch_size = x_flat.shape[0]
        
        # 确保在正确的设备上
        self.kan = self.kan.to(x.device)
        
        try:
            # 对于大批量，使用chunking
            if batch_size > self.chunk_size:
                logger.debug(f"{YELLOW}📦 使用分块处理: batch_size={batch_size}, chunk_size={self.chunk_size}{RESET}")
                out_chunks = []
                for i in range(0, batch_size, self.chunk_size):
                    chunk = x_flat[i:i+self.chunk_size]
                    out_chunk = self.kan(chunk)
                    out_chunks.append(out_chunk)
                out_flat = torch.cat(out_chunks, dim=0)
            else:
                out_flat = self.kan(x_flat)
            
            out = out_flat.reshape(*original_shape[:-1], self.out_features)
            return out
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # OOM错误处理
            self.oom_count += 1
            logger.warning(f"{YELLOW}⚠️ KAN前向传播失败 (OOM次数: {self.oom_count})，自动切换为Linear层{RESET}")
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 使用预先创建的fallback层
            return self.linear_fallback(x)


class KANNetwork(nn.Module):
    """
    多层KAN网络
    """
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 grid_size: int = 5,
                 spline_order: int = 3,
                 dropout: float = 0.1,
                 use_kan: bool = True):
        """
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            hidden_dims: 隐藏层维度列表
            grid_size: KAN网格大小
            spline_order: KAN样条阶数
            dropout: Dropout比率
            use_kan: 是否使用KAN (False则使用MLP)
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_kan = use_kan
        
        # 构建层
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            if use_kan:
                layers.append(KANLinear(
                    dims[i], 
                    dims[i+1],
                    grid_size=grid_size,
                    spline_order=spline_order
                ))
            else:
                # 标准MLP (用于消融实验)
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:  # 不在最后一层添加激活
                    layers.append(nn.SiLU())
            
            # Dropout (不在最后一层)
            if i < len(dims) - 2 and dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"KANNetwork initialized: {dims}, use_kan={use_kan}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (..., in_dim)
        
        Returns:
            输出张量 (..., out_dim)
        """
        return self.network(x)


def create_kan_or_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Optional[List[int]] = None,
    use_kan: bool = True,
    grid_size: int = 5,
    spline_order: int = 3,
    dropout: float = 0.1,
    chunk_size: int = 999999
) -> nn.Module:
    """
    工厂函数：创建KAN或MLP
    
    用于消融实验切换
    
    Args:
        chunk_size: KAN分块大小（用于节省显存）
    """
    if hidden_dims is None:
        hidden_dims = []
    
    if len(hidden_dims) == 0:
        # 单层
        if use_kan:
            return KANLinear(
                in_features=in_dim,
                out_features=out_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                chunk_size=chunk_size
            )
        else:
            return nn.Linear(in_dim, out_dim)
    else:
        # 多层
        return KANNetwork(
            in_dim, out_dim, hidden_dims,
            grid_size, spline_order, dropout, use_kan
        )
