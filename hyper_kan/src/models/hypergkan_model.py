"""
HyperGKAN 完整模型
Seq2Seq架构: Encoder-Decoder with GRU + HyperGKAN layers
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging
from .hypergkan_conv import DualHyperGKANConv

logger = logging.getLogger(__name__)


class HyperGKANLayer(nn.Module):
    """
    单个HyperGKAN时空层
    集成: 双超图卷积 + 时序建模
    """
    
    def __init__(self,
                 d_model: int,
                 hidden_channels: int,
                 use_kan: bool = True,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 kan_chunk_size: int = 999999,
                 dropout: float = 0.1,
                 activation: str = "silu",
                 fusion_method: str = "concat",
                 float32_norm: bool = True,
                 degree_clamp_min: float = 1e-6):
        super().__init__()
        
        # 双超图卷积
        self.spatial_conv = DualHyperGKANConv(
            in_channels=d_model,
            out_channels=hidden_channels,
            use_kan=use_kan,
            grid_size=grid_size,
            spline_order=spline_order,
            kan_chunk_size=kan_chunk_size,
            dropout=dropout,
            activation=activation,
            fusion_method=fusion_method,
            float32_norm=float32_norm,
            degree_clamp_min=degree_clamp_min
        )
        
        # 残差连接的投影层 (如果维度不匹配)
        if d_model != hidden_channels:
            self.residual_proj = nn.Linear(d_model, hidden_channels)
        else:
            self.residual_proj = None
        
        # Layer Normalization
        self.norm = nn.LayerNorm(hidden_channels)
    
    def forward(self,
                x: torch.Tensor,
                H_nei: torch.Tensor,
                H_sem: torch.Tensor,
                W_nei: Optional[torch.Tensor] = None,
                W_sem: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C)
        
        Returns:
            (B, T, N, C_out)
        """
        B, T, N, C = x.shape
        
        # 时间步循环处理
        outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, C)
            
            # 空间卷积
            out_t = self.spatial_conv(x_t, H_nei, H_sem, W_nei, W_sem)  # (B, N, C_out)
            
            # 残差连接
            if self.residual_proj is not None:
                residual = self.residual_proj(x_t)
            else:
                residual = x_t
            
            out_t = out_t + residual
            
            # Layer Norm
            out_t = self.norm(out_t)
            
            outputs.append(out_t)
        
        output = torch.stack(outputs, dim=1)  # (B, T, N, C_out)
        
        return output


class HyperGKAN(nn.Module):
    """
    完整的HyperGKAN模型
    
    架构:
    1. 输入投影
    2. Encoder: GRU + HyperGKAN layers
    3. Decoder: GRU + HyperGKAN layers
    4. 输出投影
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int = 64,
                 num_hypergkan_layers: int = 2,
                 hidden_channels: int = 64,
                 gru_hidden_size: int = 64,
                 gru_num_layers: int = 2,
                 use_kan: bool = True,
                 kan_grid_size: int = 5,
                 kan_spline_order: int = 3,
                 kan_chunks: int = 1,
                 dropout: float = 0.1,
                 fusion_method: str = "concat",
                 gru_type: str = "gru",
                 float32_norm: bool = True,
                 degree_clamp_min: float = 1e-6):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            d_model: 模型隐藏维度
            num_hypergkan_layers: HyperGKAN层数
            hidden_channels: 超图卷积隐藏维度
            gru_hidden_size: GRU隐藏层大小
            gru_num_layers: GRU层数
            use_kan: 是否使用KAN
            kan_grid_size: KAN网格大小
            kan_spline_order: KAN样条阶数
            kan_chunks: KAN分块数 (1=最快, 12=最省显存)
            dropout: Dropout比率
            fusion_method: 超图融合方式
            gru_type: 时序模块类型 ("gru" or "lstm")
            float32_norm: 超图卷积归一化是否使用float32 (由 element_settings.py 控制)
            degree_clamp_min: 节点度最小 clamp 值 (由 element_settings.py 控制)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.kan_chunks = kan_chunks
        
        # 根据chunks计算chunk_size (用于KAN层的分块处理)
        # 假设batch=4, time=12, stations=2048: total_samples = 4*12*2048 = 98304
        # chunks=1: chunk_size=98304 (一次性)
        # chunks=12: chunk_size=8192 (按时间步分块)
        self.kan_chunk_size = 8192 // kan_chunks if kan_chunks > 1 else 999999
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Encoder: HyperGKAN layers
        self.encoder_layers = nn.ModuleList([
            HyperGKANLayer(
                d_model=d_model if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                use_kan=use_kan,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                kan_chunk_size=self.kan_chunk_size,
                dropout=dropout,
                activation="silu",
                fusion_method=fusion_method,
                float32_norm=float32_norm,
                degree_clamp_min=degree_clamp_min
            )
            for i in range(num_hypergkan_layers)
        ])
        
        # Encoder GRU
        if gru_type == "gru":
            self.encoder_rnn = nn.GRU(
                input_size=hidden_channels,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0,
                bidirectional=False
            )
        elif gru_type == "lstm":
            self.encoder_rnn = nn.LSTM(
                input_size=hidden_channels,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0,
                bidirectional=False
            )
        else:
            raise ValueError(f"Unknown RNN type: {gru_type}")
        
        self.gru_type = gru_type
        
        # Decoder: HyperGKAN layers
        # 注意: decoder_input 来自 encoder HyperGKAN 输出 (hidden_channels 维),
        #       而非 encoder GRU 输出, 所以所有 decoder 层的 d_model 都是 hidden_channels
        self.decoder_layers = nn.ModuleList([
            HyperGKANLayer(
                d_model=hidden_channels,
                hidden_channels=hidden_channels,
                use_kan=use_kan,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                kan_chunk_size=self.kan_chunk_size,
                dropout=dropout,
                activation="silu",
                fusion_method=fusion_method,
                float32_norm=float32_norm,
                degree_clamp_min=degree_clamp_min
            )
            for i in range(num_hypergkan_layers)
        ])
        
        # Decoder GRU
        if gru_type == "gru":
            self.decoder_rnn = nn.GRU(
                input_size=hidden_channels,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0,
                bidirectional=False
            )
        elif gru_type == "lstm":
            self.decoder_rnn = nn.LSTM(
                input_size=hidden_channels,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0,
                bidirectional=False
            )
        
        # 输出投影
        self.output_projection = nn.Linear(gru_hidden_size, output_dim)
        
        # DataParallel 兼容标志
        self._use_dp_buffers = False
        
        logger.info(f"HyperGKAN initialized:")
        logger.info(f"  Input dim: {input_dim}, Output dim: {output_dim}")
        logger.info(f"  d_model: {d_model}, hidden_channels: {hidden_channels}")
        logger.info(f"  HyperGKAN layers: {num_hypergkan_layers}")
        logger.info(f"  GRU: {gru_num_layers} layers x {gru_hidden_size} hidden")
        logger.info(f"  Use KAN: {use_kan}")
    
    def set_hypergraph_buffers(self, H_nei, H_sem, W_nei=None, W_sem=None):
        """
        将超图矩阵注册为模型 buffer（用于 DataParallel 多卡训练）。
        
        DataParallel 会将 forward() 的所有 tensor 参数沿 batch 维切分，
        但超图矩阵 H_nei (N, E) 不含 batch 维，被切分后会导致结果错误。
        因此需要将超图矩阵注册为 buffer，它们会随模型自动复制到各 GPU。
        
        必须在 nn.DataParallel(model) 包装之前调用此方法。
        """
        self.register_buffer('_dp_H_nei', H_nei)
        self.register_buffer('_dp_H_sem', H_sem)
        self.register_buffer('_dp_W_nei', W_nei)
        self.register_buffer('_dp_W_sem', W_sem)
        self._use_dp_buffers = True
        logger.info("Hypergraph tensors registered as model buffers (DataParallel mode)")
    
    def forward(self,
                x: torch.Tensor,
                H_nei: Optional[torch.Tensor] = None,
                H_sem: Optional[torch.Tensor] = None,
                W_nei: Optional[torch.Tensor] = None,
                W_sem: Optional[torch.Tensor] = None,
                output_length: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (B, T_in, N, F)
            H_nei: 邻域超图 (N, E1)  — 单卡模式传入, 多卡模式从 buffer 读取
            H_sem: 语义超图 (N, E2)  — 单卡模式传入, 多卡模式从 buffer 读取
            W_nei: 邻域权重 (E1,)
            W_sem: 语义权重 (E2,)
            output_length: 输出序列长度 (如果为None则等于T_in)
        
        Returns:
            输出序列 (B, T_out, N, F_out)
        """
        # DataParallel 模式: 从模型 buffer 读取超图矩阵
        if H_nei is None and self._use_dp_buffers:
            H_nei = self._dp_H_nei
            H_sem = self._dp_H_sem
            W_nei = self._dp_W_nei
            W_sem = self._dp_W_sem
        
        B, T_in, N, F = x.shape
        
        if output_length is None:
            output_length = T_in
        
        # 输入投影
        x = self.input_projection(x)  # (B, T_in, N, d_model)
        
        # ===== Encoder =====
        # HyperGKAN layers
        for layer in self.encoder_layers:
            x = layer(x, H_nei, H_sem, W_nei, W_sem)  # (B, T_in, N, hidden_channels)
        
        # Reshape for RNN: (B*N, T_in, hidden_channels)
        x_rnn = x.permute(0, 2, 1, 3).reshape(B * N, T_in, -1)
        
        # Encoder RNN
        if self.gru_type == "gru":
            _, h_n = self.encoder_rnn(x_rnn)  # h_n: (num_layers, B*N, hidden)
        else:  # lstm
            _, (h_n, c_n) = self.encoder_rnn(x_rnn)
        
        # ===== Decoder =====
        # 使用encoder的最后状态初始化decoder
        # 自回归生成 (teacher forcing可选)
        
        # 这里使用简化版本: 直接用encoder的输出作为decoder输入
        # 在实际应用中可以使用自回归或teacher forcing
        
        # 取encoder最后一个时间步作为decoder初始输入
        decoder_input = x[:, -1:, :, :]  # (B, 1, N, hidden_channels)
        decoder_input = decoder_input.expand(B, output_length, N, -1)  # (B, T_out, N, hidden_channels)
        
        # HyperGKAN layers
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input, H_nei, H_sem, W_nei, W_sem)
        
        # Reshape for RNN
        decoder_rnn_input = decoder_input.permute(0, 2, 1, 3).reshape(B * N, output_length, -1)
        
        # Decoder RNN (使用encoder的hidden state)
        if self.gru_type == "gru":
            decoder_output, _ = self.decoder_rnn(decoder_rnn_input, h_n)
        else:  # lstm
            decoder_output, _ = self.decoder_rnn(decoder_rnn_input, (h_n, c_n))
        
        # Reshape back: (B, N, T_out, hidden) -> (B, T_out, N, hidden)
        decoder_output = decoder_output.reshape(B, N, output_length, -1).permute(0, 2, 1, 3)
        
        # 输出投影
        output = self.output_projection(decoder_output)  # (B, T_out, N, output_dim)
        
        return output
    
    def get_num_parameters(self) -> int:
        """返回模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
