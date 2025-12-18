import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf*mask, dim=1)
        x_inv = x - x_var
        
        return x_var, x_inv
        
class WaveletFilter(nn.Module):
    #Wavelet Filter: to time-variant and time-invariant term
    def __init__(self, wavelet='db4', level=1, seq_len=48):
        super(WaveletFilter, self).__init__()
        self.dwt = DWTForward(J=level, wave=wavelet, mode='zero')  
        self.idwt = DWTInverse(wave=wavelet, mode='zero')  
        self.seq_len = seq_len
        self.gate = nn.Sequential(
            nn.Conv1d(self.seq_len*2, 1, kernel_size=1),  # 输入 [x, x_inv] 拼接后通道为2
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_approch, x_detail = self.dwt(x.unsqueeze(1)) 
        mask = [torch.zeros_like(c) for c in x_detail]
        x_inv = self.idwt((x_approch, mask)).squeeze(1) 
        min_length = min(x.size(-1), x_inv.size(-1))
        x = x[..., :min_length] 
        x_inv = x_inv[..., :min_length] 

        x_cat = torch.cat([x, x_inv], dim=1)  # 拼接后形状为 [B, 2*T, L]
        g = self.gate(x_cat)
        g = g.expand(-1, self.seq_len, -1)
       
        x_var = g * (x - x_inv)
        x_inv_new = x - x_var

        return x_var, x_inv_new

class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 dropout=0.05,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
    

class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """
    def __init__(self): 
        super(KPLayer, self).__init__()
        
        self.K = None # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            #print("BBBBBB")
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred
    
    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred= self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """
    def __init__(self): 
        super(KPLayerApprox, self).__init__()
        
        self.K = None # B E E
        self.K_step = None # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            #print("AAAAAAA")
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1) # B L E
        
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred
    

class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder            
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer() 

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L-self.padding_len:, :], x) ,dim=1)

        res = res.chunk(self.freq, dim=1)     # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)   # B F PC

        res = self.encoder(res) # B F H
        x_rec, x_pred = self.dynamics(res, self.step) # B F H, B S H

        x_rec = self.decoder(x_rec) # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
        
        x_pred = self.decoder(x_pred)     # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :] # B S C

        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """
    def __init__(self,
                 input_len=96,
                 pred_len=96,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init) # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())
    
    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2) # B C L
        res = self.encoder(res) # B C H
        res = self.K(res) # B C H
        res = self.decoder(res) # B C S
        res = res.transpose(1, 2) # B S C

        return res

def full_attention_conv(query, key, value, kernel='simple', output_attn=False):
    """
    query: [batch_size, num_nodes, num_heads, hidden_channels]
    key: [batch_size, num_nodes, num_heads, hidden_channels]
    value: [batch_size, num_nodes, num_heads, hidden_channels]
    kernel: Choice of attention kernel ('simple', 'sigmoid', etc.)
    output_attn: Whether to return attention weights

    return:
    attn_output: [batch_size, hidden_channels, num_nodes]
    """
    batch_size, num_nodes, num_heads, hidden_channels = query.shape
    
    if kernel == 'simple':
        # Normalize input
        query = query / torch.norm(query, p=2, dim=-1, keepdim=True)
        key = key / torch.norm(key, p=2, dim=-1, keepdim=True)

        # Numerator
        kvs = torch.einsum("bnhm,bnhd->bhmd", key, value)  # [batch, head, hidden, hidden]
        attention_num = torch.einsum("bnhm,bhmd->bnhd", query, kvs)  # [batch, num_nodes, num_heads, hidden_channels]
        
        all_ones = torch.ones([num_nodes], device=value.device)
        vs_sum = torch.einsum("bnhd,n->bhd", value, all_ones)  # [batch, head, hidden]
        attention_num += vs_sum.unsqueeze(1)  # [batch, num_nodes, num_heads, hidden_channels]

        # Denominator
        ks_sum = torch.einsum("bnhm,n->bhm", key, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", query, ks_sum)  # [batch, num_nodes, num_heads]
        attention_normalizer = attention_normalizer.unsqueeze(-1) + num_nodes  # [batch, num_nodes, num_heads, 1]
        
        attn_output = attention_num / attention_normalizer  # [batch, num_nodes, num_heads, hidden_channels]

        if output_attn:
            attention = torch.einsum("bnhm,bnhm->bnhn", query, key) / attention_normalizer  # [batch, num_nodes, num_heads, num_nodes]
    
    elif kernel == 'sigmoid':
        # Compute attention scores
        attention_num = torch.sigmoid(torch.einsum("bnhm,bnhm->bnhn", query, key))  # [batch, num_nodes, num_heads, num_nodes]
        
        all_ones = torch.ones([num_nodes], device=key.device)
        attention_normalizer = torch.einsum("bnhn,n->bnh", attention_num, all_ones).unsqueeze(1)  # [batch, 1, num_heads, num_nodes]
        attention_normalizer = attention_normalizer.expand(-1, num_nodes, -1, -1)  # [batch, num_nodes, num_heads, num_nodes]
        
        attention = attention_num / attention_normalizer
        attn_output = torch.einsum("bnhn,bnhd->bnhd", attention, value)  # [batch, num_nodes, num_heads, hidden_channels]
    
    # Reshape output to match target shape [batch_size, hidden_channels, num_nodes]
    attn_output = attn_output.permute(0, 3, 1, 2).contiguous()  # [batch, hidden_channels, num_nodes, num_heads]
    attn_output = attn_output.mean(dim=-1)  # Aggregate over heads -> [batch, hidden_channels, num_nodes]
    
    if output_attn:
        return attn_output, attention
    else:
        return attn_output

def gcn_conv(x, adj):
    #输入：x:[batch_size, num_nodes, num_heads, hidden_channels]
    #输出：x:[batch_size, hidden_channels, num_nodes]
    batch_size, num_nodes, num_heads, hidden_channels = x.shape
    
    # 将邻接矩阵转为稀疏表示
    adj = adj + torch.eye(num_nodes).to(adj.device)  # 加上自连接（单位矩阵），确保每个节点至少与自己相连
    adj = adj / adj.sum(dim=1, keepdim=True)  # 归一化邻接矩阵
    # 对每个头部进行图卷积
    gcn_conv_output = []
    for i in range(num_heads):
        x_i = x[:, :, i, :]  # [batch_size, num_nodes, hidden_channels]
        
        # 对每个批次执行图卷积，邻接矩阵与节点特征矩阵相乘
        output_i = torch.matmul(adj, x_i)  # [batch_size, num_nodes, hidden_channels]
        
        # 保存每个头部的输出
        gcn_conv_output.append(output_i)

    # 将每个头部的结果结合
    gcn_conv_output = torch.mean(torch.stack(gcn_conv_output, dim=1), dim=1)  # [batch_size, num_nodes, hidden_channels]
    gcn_conv_output = gcn_conv_output.permute(0, 2, 1)  # [batch_size, hidden_channels, num_nodes]
    return gcn_conv_output

class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads,
               kernel='simple',
               use_graph=True,
               use_weight=True):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_graph = use_graph
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, adj, output_attn=False):
         #输入：x：[batch_size,hidden_channels, num_nodes ]；输出：x: [batch_size,num_nodes,hidden_channels]
        """
        query_input: [batch_size,hidden_channels, num_nodes ]
        source_input:[batch_size,hidden_channels, num_nodes ]
        adj: [num_nodes, num_nodes]
        output_attn: 是否输出注意力权重
        """
        batch_size, hidden_channels, num_nodes = query_input.shape
        # 1. 交换维度以适配 nn.Linear，变为 [batch_size, num_nodes, hidden_channels]
        query_input = query_input.permute(0, 2, 1)  # [batch_size, num_nodes, hidden_channels]
        source_input = source_input.permute(0, 2, 1)  # [batch_size, num_nodes, hidden_channels]
        #'''
        # 2. 线性变换，计算 Query、Key、Value
        query = self.Wq(query_input).reshape(batch_size, num_nodes, self.num_heads, self.out_channels) # [batch_size, num_nodes, num_heads, hidden_channels]
        key = self.Wk(source_input).reshape(batch_size, num_nodes, self.num_heads, self.out_channels) # [batch_size, num_nodes, num_heads, hidden_channels]
        if self.use_weight:
            value = self.Wv(source_input).reshape(batch_size, num_nodes, self.num_heads, self.out_channels) # [batch_size, num_nodes, num_heads, hidden_channels]
        else:
            value = source_input.unsqueeze(2)  # [batch_size, num_nodes, 1, hidden_channels]
        # 3. 计算注意力
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, self.kernel, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query,key,value,self.kernel) # [N, H, D]
        
        # 4. 计算 GCN（适配批量处理）
        if self.use_graph:
            gcn_conv_output = gcn_conv(value,adj)  # [batch_size, num_nodes, hidden_channels]
            final_output = attention_output + gcn_conv_output
        else:
            final_output = attention_output  #输出：x: [batch_size,num_nodes,hidden_channels]
        if output_attn:
            return final_output, attn
        else:
            return final_output

class DIFFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, kernel='simple',
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True):
        super(DIFFormer, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                DIFFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel=kernel, use_graph=use_graph, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, adj):
        #输入x：[batch_size, in_features, num_nodes ]
        layer_ = []
        x = x.permute(0, 2, 1)  # x：[batch_size, num_nodes, in_features]
        # input MLP layer
        x = self.fcs[0](x)   # x：[batch_size, num_nodes, hidden_channels]
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)    

        # store as residual link
        layer_.append(x)
       
        for i, conv in enumerate(self.convs):
            x = x.permute(0, 2, 1)
            # graph convolution with DIFFormer layer
            x = conv(x, x, adj)  #输入：x：[batch_size, hidden_channels, num_nodes]；输出：x: [batch_size,num_nodes,hidden_channels]
            if self.residual:
                x = x.permute(0, 2, 1)
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)  
            layer_.append(x)

        # output MLP layer
        x_out = self.fcs[-1](x)     #x_out：[batch_size, num_nodes, in_features]
        x_out = x_out.permute(0,2,1) #x_out：[batch_size, in_features, num_nodes ]
        return x_out    

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0) # [layer num, N, N]


class Model(nn.Module):
    '''
    Koopman Forecasting Model
    '''
    def __init__(self, configs, adj):
        super(Model, self).__init__()
        self.mask_spectrum = configs.mask_spectrum
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_layers = configs.hidden_layers
        self.multistep = configs.multistep
        self.gcn_hidden_num = configs.gcn_hidden_num
        self.hidden_num = configs.hidden_num
        self.batch_size = configs.batch_size
        self.gcn_alpha = configs.gcn_alpha
        self.mixup_alpha = configs.mixup_alpha
        self.diff_hidden_channels = configs.diff_hidden_channels
        self.diff_dropout = configs.diff_dropout
        self.diff_num_heads = configs.diff_num_heads
        self.wavelet = configs.wavelet
        self.wavelet_level = configs.wavelet_level

        #Filter
        self.select_filter = configs.select_filter
        if self.select_filter == 0:
            self.disentanglement = FourierFilter(self.mask_spectrum)
        elif self.select_filter == 1:
            self.disentanglement = WaveletFilter(wavelet=self.wavelet, level=self.wavelet_level, seq_len=self.input_len)
        self.register_buffer("adj", adj)
        #DIFFormer
        self.difformer = DIFFormer(in_channels=self.input_len, hidden_channels=self.diff_hidden_channels, out_channels=self.input_len, num_layers=2, num_heads=self.diff_num_heads, kernel='simple', alpha=0.5, dropout=self.diff_dropout, use_bn=True, use_residual=True, use_weight=True, use_graph=False)

        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_kps = self.time_var_kps = nn.ModuleList([
                                TimeInvKP(input_len=self.input_len,
                                    pred_len=self.pred_len, 
                                    dynamic_dim=self.dynamic_dim,
                                    encoder=self.time_inv_encoder, 
                                    decoder=self.time_inv_decoder)
                                for _ in range(self.num_blocks)])

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
                    TimeVarKP(enc_in=configs.enc_in,
                        input_len=self.input_len,
                        pred_len=self.pred_len,
                        seg_len=self.seg_len,
                        dynamic_dim=self.dynamic_dim,
                        encoder=self.time_var_encoder,
                        decoder=self.time_var_decoder,
                        multistep=self.multistep)
                    for _ in range(self.num_blocks)])
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: B L C

        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            #Filter
            time_var_input, time_inv_input = self.disentanglement(residual)
            #DIFFormer
            time_inv_input = self.difformer(time_inv_input, self.adj)
            time_var_input = self.difformer(time_var_input, self.adj)

            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = (time_inv_output + time_var_output)
            else:
                forecast += (time_inv_output + time_var_output)

        # Series Stationarization adopted from NSformer
        res = forecast * std_enc + mean_enc

        return res
