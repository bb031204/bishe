# File: STGNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

################################################################################
# Basic Layers (从model.py中复制核心组件)
################################################################################

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
    def forward(self, x):        return self.mlp(x)

class nconv(nn.Module):
    def __init__(self):        super().__init__()
    def forward(self, x, A):    return torch.einsum('ncwl,vw->ncvl', x, A).contiguous()

class MixHop(nn.Module):
    def __init__(self, c_in, c_out, hop, hopalpha, fusion):
        super().__init__()
        self.nconv = nconv()
        self.hop = hop
        self.alpha = hopalpha
        self.fusion = fusion
        if fusion == 'concat':  self.out = linear((hop+1)*c_in, c_out)
        elif fusion in ['sum', 'mean', 'max']: self.out = linear(c_in, c_out)
        self.mlp = nn.ModuleList([linear(c_in, c_out) for _ in range(hop)])
        
    def forward(self, x, adj):
        adj = adj + torch.eye(adj.shape[0]).to(x.device)
        d = adj.sum(1)
        a = adj / d.view(-1, 1)
        h = x
        out = [h]
        for i in range(self.hop):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
            out.append(self.mlp[i](h))
        if self.fusion == 'concat': return self.out(torch.cat(out, dim=1))
        elif self.fusion == 'sum':  return self.out(torch.sum(torch.stack(out), dim=0))
        else: raise ValueError('Invalid fusion type')

class S_MixHop(nn.Module):
    def __init__(self, cin, cout, hop, hopalpha, fusion, G_mix):
        super().__init__()
        self.gconvf1 = MixHop(cin, cout, hop, hopalpha, fusion)
        self.gconvf2 = MixHop(cin, cout, hop, hopalpha, fusion)
        self.gconvb1 = MixHop(cin, cout, hop, hopalpha, fusion)
        self.gconvb2 = MixHop(cin, cout, hop, hopalpha, fusion)
        self.G_mix = G_mix
        
    def forward(self, x, adj):
        x0 = self.gconvf1(x, adj[0]) + self.gconvb1(x, adj[0].T)
        x1 = self.gconvf2(x, adj[1]) + self.gconvb2(x, adj[1].T)
        return self.G_mix * x1 + (1 - self.G_mix) * x0

class T_Inception(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor, dropout):
        super().__init__()
        assert cout % len(kernel_set) == 0
        self.kernel_set = kernel_set
        self.tconv_f = nn.ModuleList()
        self.tconv_g = nn.ModuleList()
        cout = cout // len(kernel_set)
        for kern in kernel_set:
            self.tconv_f.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
            self.tconv_g.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        f, g = [], []
        for i in range(len(self.kernel_set)):
            f.append(self.tconv_f[i](x))
            g.append(self.tconv_g[i](x))
        f = torch.cat([t[..., -f[-1].size(3):] for t in f], dim=1)
        g = torch.cat([t[..., -g[-1].size(3):] for t in g], dim=1)
        return self.do(torch.tanh(f) * torch.sigmoid(g))

class G_MTGNN(nn.Module):
    def __init__(self, N, k, dim, alpha=3):
        super().__init__()
        self.N = N
        self.emb1 = nn.Embedding(N, dim)
        self.emb2 = nn.Embedding(N, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.k = k
        self.dim = dim
        self.alpha = alpha
        
    def forward(self, idx):
        nodevec1 = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        nodevec2 = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        a = torch.mm(nodevec1, nodevec2.T) - torch.mm(nodevec2, nodevec1.T)
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros_like(adj)
        s1, t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        return adj * mask

################################################################################
# STGNN Core Model (适配到框架)
################################################################################

class STGNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # 从数据特征获取参数
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.adj = torch.Tensor(data_feature['adj_mx']).to(config['device'])
        self.idx = torch.arange(self.num_nodes).to(config['device'])
        
        # 从配置获取参数
        self.device = config.get('device', torch.device('cpu'))
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()
        self.config = config
        
        # 初始化模型组件
        self.gc = G_MTGNN(
            self.num_nodes, 
            config.get('G_k', 3), 
            config.get('G_dim', 10), 
            config.get('G_alpha', 3)
        )
        self._build_model()
        self._init_parameters()

    def _build_model(self):
        # 核心参数
        self.mid_c = self.config.get('mid_channel', 64)
        self.skip_c = self.config.get('skip_channel', 64)
        self.end_c = self.config.get('end_channel', 64)
        self.dropout = self.config.get('dropout', 0.3)
        self.routing = self.config.get('R_routing', ['T', 'S', 'T'])
        
        # 网络结构
        self.start_conv = linear(self.feature_dim, self.mid_c)
        self.blocks = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.skipT = nn.ModuleList()
        self.skipS = nn.ModuleList()

        input_window = self.config.get('input_window', 12)
        max_ks = max(self.config.get('T_ks', [2,3,4,5]))  # 修改默认值避免kernel过大
        dilation = self.config.get('T_dilation', 1)
        self.receptive_field = 1 + len([b for b in self.routing if b == 'T']) * (max_ks-1)*dilation

        # 安全检查：确保receptive_field不超过input_window
        if self.receptive_field > input_window:
            self._logger.warning(f"Receptive field {self.receptive_field} > input_window {input_window}, clamping...")
            self.receptive_field = input_window
        
        # 构建模块
        rf_curr = 1
        # 跟踪实际的时间步维度（T块会改变，S块不改变）
        actual_time_steps = input_window
        for i, blk in enumerate(self.routing):
            if blk == 'T':
                # T块会改变时间步维度：input_time - max_kernel + 1
                # 先更新时间步，因为skipT是在T块之后应用的
                actual_time_steps = actual_time_steps - max_ks + 1
                # 安全检查
                if actual_time_steps < 1:
                    raise ValueError(f"Time steps reduced to {actual_time_steps}, "
                                     f"input_window={input_window}, max_ks={max_ks}, "
                                     f"consider reducing max_ks or number of T blocks")
                self.blocks.append(T_Inception(
                    self.mid_c, self.mid_c,
                    self.config['T_ks'], dilation, self.dropout
                ))
                # skipT的kernel size应该等于T块之后的时间步维度
                self.skipT.append(nn.Conv2d(
                    self.mid_c, self.skip_c,
                    kernel_size=(1, actual_time_steps)
                ))
                rf_curr += (max_ks-1)*dilation
            elif blk == 'S':
                self.blocks.append(S_MixHop(
                    self.mid_c, self.mid_c,
                    self.config.get('S_hop', 2),
                    self.config.get('S_hopalpha', 0.5),
                    self.config.get('S_fusion', 'concat'),
                    self.config.get('G_mix', 0.5)
                ))
                # skipS的kernel size应该等于当前的时间步维度
                self.skipS.append(nn.Conv2d(
                    self.mid_c, self.skip_c,
                    kernel_size=(1, actual_time_steps)
                ))
                # S块不改变时间步维度，使用actual_time_steps
                self.norm.append(nn.LayerNorm([self.mid_c, self.num_nodes, actual_time_steps]))
        
        # 输出层
        self.end_conv_1 = linear(self.skip_c, self.end_c)
        self.end_conv_2 = linear(self.end_c, self.output_dim)
        # 安全检查：确保skipInput的kernel_size不超过input_window
        skip_kernel_size = min(self.config['input_window'], self.receptive_field)
        self.skipInput = nn.Conv2d(
            self.feature_dim, self.skip_c,
            kernel_size=(1, skip_kernel_size)
        )

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.uniform_(p)

    def forward(self, batch):
        x = batch['X'].permute(0, 3, 2, 1)  # (B, T, N, F) -> (B, F, N, T) 修复维度顺序
        adj = [self.adj, self.gc(self.idx)]  # 组合GT和动态生成的邻接矩阵

        # 前向传播
        skip_input = self.skipInput(F.dropout(x, self.dropout))
        # 将skip_input的时间维度(最后一维)压缩 - 使用mean对时间维度求平均
        skip_input = skip_input.mean(dim=-1)  # (B, skip_c, N, T) -> (B, skip_c, N)

        x = self.start_conv(x)
        skip_list = []

        for i, blk in enumerate(self.routing):
            if blk == 'T':
                x = self.blocks[i](x)
                skip_t = self.skipT[len(self.skipT)-1](x)
                # 将时间维度(最后一维)压缩 - 使用mean对时间维度求平均
                skip_t = skip_t.mean(dim=-1)  # (B, skip_c, N, T) -> (B, skip_c, N)
                skip_list.append(skip_t)
            elif blk == 'S':
                x = self.blocks[i](x, adj)
                x = self.norm[len(self.norm)-1](x)
                skip_s = self.skipS[len(self.skipS)-1](x)
                # 将时间维度(最后一维)压缩 - 使用mean对时间维度求平均
                skip_s = skip_s.mean(dim=-1)  # (B, skip_c, N, T) -> (B, skip_c, N)
                skip_list.append(skip_s)

        # 输出处理 - 现在所有skip都是(B, skip_c, N)维度，可以安全相加
        skip = sum(skip_list) + skip_input  # (B, skip_c, N)

        # end_conv_1需要4D输入，需要添加一个维度
        skip = skip.unsqueeze(-1)  # (B, skip_c, N, 1)
        x = F.relu(self.end_conv_1(F.relu(skip)))  # (B, end_c, N, 1)
        x = self.end_conv_2(x)  # (B, output_dim, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, N, output_dim)

        # 扩展到output_window时间步 (复制output_window-1次)
        output_window = self.config.get('output_window', 12)
        x = x.unsqueeze(1).repeat(1, output_window, 1, 1)  # (B, output_window, N, output_dim)

        return x[..., :self.output_dim]

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_pred = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
        return loss.masked_mae_torch(y_pred, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)