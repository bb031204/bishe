"""
HyperGKAN 超图卷积层
基于KAN的超图卷积

数值稳定性参数由 element_settings.py 控制（每个数据集可独立配置）：
  - float32_norm: 归一化矩阵是否强制使用 float32 (防止 AMP float16 下溢)
  - degree_clamp_min: 节点/超边度的最小值 (防止零度节点导致 Inf/NaN)
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging
from .kan_layer import create_kan_or_mlp

logger = logging.getLogger(__name__)


class HyperGKANConv(nn.Module):
    """
    KAN-based 超图卷积层
    
    实现论文中的公式:
    X^l = Φ_l(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{l-1})
    
    其中 Φ_l 是KAN而非传统的线性变换
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_kan: bool = True,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 kan_chunk_size: int = 999999,
                 dropout: float = 0.1,
                 activation: str = "silu",
                 float32_norm: bool = True,
                 degree_clamp_min: float = 1e-6):
        """
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            use_kan: 是否使用KAN (False则使用MLP)
            grid_size: KAN网格大小
            spline_order: KAN样条阶数
            kan_chunk_size: KAN层分块大小
            dropout: Dropout比率
            activation: 激活函数类型
            float32_norm: 是否强制使用 float32 计算归一化矩阵
                          (由 element_settings.py 中各数据集的 numerical.conv_float32_norm 控制)
            degree_clamp_min: 节点/超边度的最小 clamp 值
                              (由 element_settings.py 中各数据集的 numerical.degree_clamp_min 控制)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_kan = use_kan
        self.float32_norm = float32_norm
        self.degree_clamp_min = degree_clamp_min
        
        # 特征变换层 (KAN或MLP)
        self.transform = create_kan_or_mlp(
            in_dim=in_channels,
            out_dim=out_channels,
            hidden_dims=[],  # 单层
            use_kan=use_kan,
            grid_size=grid_size,
            spline_order=spline_order,
            chunk_size=kan_chunk_size,
            dropout=0.0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        logger.debug(
            f"HyperGKANConv: {in_channels} -> {out_channels}, use_kan={use_kan}, "
            f"float32_norm={float32_norm}, degree_clamp_min={degree_clamp_min}"
        )
    
    def forward(self,
                x: torch.Tensor,
                H: torch.Tensor,
                W: Optional[torch.Tensor] = None,
                D_v: Optional[torch.Tensor] = None,
                D_e: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        超图卷积前向传播
        
        Args:
            x: 节点特征 (B, N, C_in) 或 (N, C_in)
            H: 关联矩阵 (N, E)
            W: 超边权重 (E,) 或 None
            D_v: 节点度矩阵 (N, N) 或 None (自动计算)
            D_e: 超边度矩阵 (E, E) 或 None (自动计算)
        
        Returns:
            输出特征 (B, N, C_out) 或 (N, C_out)
        """
        # 判断是否有batch维度
        has_batch = (x.dim() == 3)
        
        if has_batch:
            B, N, C_in = x.shape
        else:
            N, C_in = x.shape
            x = x.unsqueeze(0)  # (1, N, C_in)
            B = 1
        
        # 确保H在正确的设备上
        H = H.to(x.device)
        E = H.shape[1]
        
        # =====================================================================
        # 数值精度策略（由 element_settings.py 的 numerical 配置驱动）
        # 当 float32_norm=True 时：
        #   归一化矩阵强制使用 float32，避免 AMP float16 下溢。
        #   科学依据：float16 最小正数 ~5.96e-8，clamp(d, min=1e-8) 会下溢为 0，
        #   导致 0^(-0.5)=Inf → NaN。对语义超图零度节点多的数据集（如 Cloud）必须开启。
        # 当 float32_norm=False 时：
        #   使用输入 x 的原始 dtype 计算（适用于确认无零度节点的数据集）。
        # =====================================================================
        original_dtype = x.dtype
        compute_dtype = torch.float32 if self.float32_norm else original_dtype
        
        H_compute = H.to(compute_dtype)
        
        # 计算度矩阵 (如果未提供)
        if D_v is None or D_e is None:
            # 处理权重
            if W is not None:
                W = W.to(x.device)
                if W.dim() == 1:
                    W_diag = W.to(compute_dtype)
                else:
                    W_diag = W.to(compute_dtype).diag()
            else:
                W_diag = torch.ones(E, device=x.device, dtype=compute_dtype)
            
            # 节点度
            d_v = H_compute @ W_diag  # (N,)
            d_v = torch.clamp(d_v, min=self.degree_clamp_min)
            
            # 超边度
            d_e = H_compute.sum(dim=0)  # (E,)
            d_e = torch.clamp(d_e, min=self.degree_clamp_min)
        else:
            d_v = D_v.to(compute_dtype).diag()
            d_e = D_e.to(compute_dtype).diag()
        
        # 归一化向量（使用向量而非对角矩阵以节省内存）
        d_v_inv_sqrt = d_v ** (-0.5)  # (N,)
        d_e_inv = d_e ** (-1.0)       # (E,)
        
        # 处理权重向量
        if W is not None:
            W_vec = W.to(x.device).to(compute_dtype)
            if W_vec.dim() > 1:
                W_vec = W_vec.diag()
        else:
            W_vec = torch.ones(E, device=x.device, dtype=compute_dtype)
        
        # 超图卷积: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X
        # 使用向量逐元素乘法代替对角矩阵乘法，更快且更省内存
        
        x_compute = x.to(compute_dtype)
        
        # 批量处理
        output_list = []
        for b in range(B):
            x_b = x_compute[b]  # (N, C_in)
            
            # D_v^{-1/2} X: 逐行乘以 d_v_inv_sqrt
            agg1 = x_b * d_v_inv_sqrt.unsqueeze(-1)  # (N, C_in)
            
            # H^T @ (D_v^{-1/2} X)
            agg2 = H_compute.T @ agg1  # (E, C_in)
            
            # D_e^{-1}: 逐行乘以 d_e_inv
            agg3 = agg2 * d_e_inv.unsqueeze(-1)  # (E, C_in)
            
            # W: 逐行乘以权重
            agg4 = agg3 * W_vec.unsqueeze(-1)  # (E, C_in)
            
            # H @ (W D_e^{-1} H^T D_v^{-1/2} X)
            agg5 = H_compute @ agg4  # (N, C_in)
            
            # D_v^{-1/2}: 再次逐行归一化
            agg6 = agg5 * d_v_inv_sqrt.unsqueeze(-1)  # (N, C_in)
            
            output_list.append(agg6)
        
        # 合并batch, 转回原始 dtype
        aggregated = torch.stack(output_list, dim=0).to(original_dtype)  # (B, N, C_in)
        
        # KAN/MLP变换
        transformed = self.transform(aggregated)  # (B, N, C_out)
        
        # 激活 + Dropout
        output = self.dropout(self.activation(transformed))
        
        # 移除batch维度(如果原本没有)
        if not has_batch:
            output = output.squeeze(0)  # (N, C_out)
        
        return output


class DualHyperGKANConv(nn.Module):
    """
    双超图卷积层 (邻域 + 语义)
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_kan: bool = True,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 kan_chunk_size: int = 999999,
                 dropout: float = 0.1,
                 activation: str = "silu",
                 fusion_method: str = "concat",
                 float32_norm: bool = True,
                 degree_clamp_min: float = 1e-6):
        """
        Args:
            fusion_method: 融合方式 ("concat", "add", "attention")
            kan_chunk_size: KAN层分块大小
            float32_norm: 由 element_settings.py 控制，传递到 HyperGKANConv
            degree_clamp_min: 由 element_settings.py 控制，传递到 HyperGKANConv
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # 邻域超图卷积
        self.conv_nei = HyperGKANConv(
            in_channels, out_channels,
            use_kan, grid_size, spline_order,
            kan_chunk_size, dropout, activation,
            float32_norm=float32_norm,
            degree_clamp_min=degree_clamp_min
        )
        
        # 语义超图卷积
        self.conv_sem = HyperGKANConv(
            in_channels, out_channels,
            use_kan, grid_size, spline_order,
            kan_chunk_size, dropout, activation,
            float32_norm=float32_norm,
            degree_clamp_min=degree_clamp_min
        )
        
        # 融合层
        if fusion_method == "concat":
            # 拼接后降维
            self.fusion = nn.Linear(out_channels * 2, out_channels)
        elif fusion_method == "attention":
            # 注意力融合
            self.attn = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.Tanh(),
                nn.Linear(out_channels, 2),
                nn.Softmax(dim=-1)
            )
        else:
            # 直接相加
            self.fusion = None
        
        logger.debug(f"DualHyperGKANConv: fusion={fusion_method}")
    
    def forward(self,
                x: torch.Tensor,
                H_nei: torch.Tensor,
                H_sem: torch.Tensor,
                W_nei: Optional[torch.Tensor] = None,
                W_sem: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 节点特征 (B, N, C_in)
            H_nei: 邻域超图 (N, E1)
            H_sem: 语义超图 (N, E2)
            W_nei: 邻域权重 (E1,)
            W_sem: 语义权重 (E2,)
        
        Returns:
            融合后的特征 (B, N, C_out)
        """
        # 邻域超图卷积
        x_nei = self.conv_nei(x, H_nei, W_nei)  # (B, N, C_out)
        
        # 语义超图卷积
        x_sem = self.conv_sem(x, H_sem, W_sem)  # (B, N, C_out)
        
        # 融合
        if self.fusion_method == "concat":
            x_fused = torch.cat([x_nei, x_sem], dim=-1)  # (B, N, 2*C_out)
            output = self.fusion(x_fused)  # (B, N, C_out)
        
        elif self.fusion_method == "attention":
            x_concat = torch.cat([x_nei, x_sem], dim=-1)  # (B, N, 2*C_out)
            attn_weights = self.attn(x_concat)  # (B, N, 2)
            
            # 加权求和
            output = attn_weights[..., 0:1] * x_nei + attn_weights[..., 1:2] * x_sem
        
        else:  # add
            output = x_nei + x_sem
        
        return output
