from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import math


# Transformer 部分
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k,n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_head = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        #scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self,device,n_heads,d_model,d_k,d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.device = device
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention(self.d_k, self.n_heads)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,device,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual) # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self,device,n_heads,d_model,d_k,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(device,n_heads,d_model,d_k,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(device,d_model,d_ff)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self,device,n_layers,n_heads,d_model,d_k,d_v,d_ff):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(device,n_heads,d_model,d_k,d_v,d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = enc_inputs
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs = layer(enc_outputs)
        return enc_outputs

class RegionAttn(nn.Module):
    def __init__(self, device,n_heads,d_model,d_k,d_v,d_ff,dataset='HUSM'):
        super().__init__()
        if dataset == 'HUSM':
            self.Reg2Chan = {'frontal':[0,1,5,8,9,10,14], 'central':[2,11,17],'parietal':[3,12,18],'occipital':[4,6,7,13,15,16]}
        elif dataset == 'LANZHOU':
            self.Reg2Chan = {'frontal':[126,125,20,13,24,21,14,8,7,127,31,25,22,17,15,9,2,1,0,124,37,32,26,23,18,10,3,123,122,121,120],\
                'central':[38,33,27,19,11,4,117,116,115,39,34,28,12,5,111,110,109,108,114,40,35,29,6,105,104,103,102,44,45,46,41,36,30,79,86,92,97,101,107],\
                    'parietal':[49,50,51,52,53,54,78,85,91,96,100,57,58,59,60,77,84,90,95,64,65,66,61,76,83,89,69,70,71,75,82,74],\
                        'occipital':[47,42,43,48,55,56,63,62,68,67,73,72,81,80,87,88,93,94,98,99,106,112,113,119,118]}
            
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.rsa = MultiHeadAttention(device,n_heads=n_heads,d_model=d_model,d_k=d_k,d_v=d_v)
        self.lsa = MultiHeadAttention(device,n_heads=n_heads,d_model=d_model,d_k=d_k,d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(device,d_model=d_model,d_ff=d_ff)
            
    def forward(self,x):
        # [ batch_size, chan, dim]
        fe_frontal = self.avgpool(x[:, self.Reg2Chan['frontal'], :].transpose(1,2))
        fe_central = self.avgpool(x[:, self.Reg2Chan['central'], :].transpose(1,2))
        fe_parietal = self.avgpool(x[:, self.Reg2Chan['parietal'], :].transpose(1,2))
        fe_occipital = self.avgpool(x[:, self.Reg2Chan['occipital'], :].transpose(1,2))
        
        fe_region = torch.concat((fe_frontal,fe_central,fe_parietal,fe_occipital),dim=2).transpose(1,2)
        # [batch_size, 4, dim]
        refined_region = self.rsa(fe_region, fe_region, fe_region)
        
        x[:, self.Reg2Chan['frontal'], :] = x[:, self.Reg2Chan['frontal'], :] + refined_region[:, 0, :].unsqueeze(1)
        x[:, self.Reg2Chan['central'], :] = x[:, self.Reg2Chan['central'], :] + refined_region[:, 1, :].unsqueeze(1)
        x[:, self.Reg2Chan['parietal'], :] = x[:, self.Reg2Chan['parietal'], :] + refined_region[:, 2, :].unsqueeze(1)
        x[:, self.Reg2Chan['occipital'], :] = x[:, self.Reg2Chan['occipital'], :] + refined_region[:, 3, :].unsqueeze(1)
        
        fused = self.lsa(x, x, x)
        fused = self.pos_ffn(fused)
        # [ batch_size, dim]
        
        return fused


class down_sample(nn.Module):
    def __init__(self, inc, kernel_size, stride, padding):
        super(down_sample, self).__init__()
        self.conv = nn.Conv2d(in_channels = inc, out_channels = inc, kernel_size = (1, kernel_size), stride = (1, stride), padding = (0, padding), bias = False)
        self.bn = nn.BatchNorm2d(inc) 
        self.elu = nn.ELU(inplace = False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.elu(self.bn(self.conv(x)))
        return output
    
class input_layer(nn.Module):
    def __init__(self, outc, groups):
        super(input_layer, self).__init__()
        self.conv_input = nn.Conv2d(in_channels = 1, out_channels = outc, kernel_size = (1, 3), 
                                    stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn_input = nn.BatchNorm2d(outc) 
        self.elu = nn.ELU(inplace = False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.bn_input(self.conv_input(x))
        return output

class Residual_Block(nn.Module): 
    def __init__(self, inc, outc, groups = 1):
        super(Residual_Block, self).__init__()
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = 1, 
                                       stride = 1, padding = 0, groups = groups, bias = False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = (1, 3), 
                               stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = (1, 3), 
                               stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.elu = nn.ELU(inplace = False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x
        output = self.bn1(self.conv1(x))
        output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

def embedding_network(input_block, Residual_Block, num_of_layer, outc, groups = 1):
    layers = []
    layers.append(input_block(outc,groups=groups))
    for i in range(0, num_of_layer):
        layers.append(Residual_Block(inc = int(math.pow(2, i)*outc), outc = int(math.pow(2, i+1)*outc),
                                     groups = groups))
    return nn.Sequential(*layers) 

class Multi_Scale_Temporal_Block(nn.Module):
    def __init__(self, outc, num_of_layer = 1):
        super().__init__() 
        self.num_of_layer = num_of_layer
        self.embedding = embedding_network(input_layer, Residual_Block, num_of_layer = num_of_layer, outc = outc, groups=1)    

        self.downsampled1 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 4, 4, 0)
        self.downsampled2 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 8, 8, 0)
        self.downsampled3 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 16, 16, 0)
        self.downsampled4 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 32, 32, 0)
        self.downsampled5 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 32, 32, 0)

    def forward(self, x):
        # [ batch_size, 1, 19, 256]
        embedding_x = self.embedding(x)#([128, 8, 15, 1280])
        # print("2",embedding_x.shape)
        cat_x = torch.cat((embedding_x, x), 1)#([128, 9, 15, 1280])
        # print("3",cat_x.shape)

        downsample1 = self.downsampled1(cat_x)#([128, 9, 15, 320])
        downsample2 = self.downsampled2(cat_x)#([128, 9, 15, 160])
        downsample3 = self.downsampled3(cat_x)#([128, 9, 15, 80])
        downsample4 = self.downsampled4(cat_x)#([128, 9, 15, 40])
        downsample5 = self.downsampled5(cat_x)#([128, 9, 15, 40])

        # print("downsample1: ", downsample1.shape)
        # print("downsample2: ", downsample2.shape)
        # print("downsample3: ", downsample3.shape)
        # print("downsample4: ", downsample4.shape)
        # print("downsample5: ", downsample5.shape)
        temporal_fe = torch.concat((downsample1,downsample2,downsample3,downsample4,downsample5),3)
        #temporal_fe = temporal_fe.transpose(1,2).contiguous()

        return temporal_fe


class Temporal_Block(nn.Module):
    def __init__(self):
        super().__init__() 
        self.mstblock1 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock2 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock3 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock4 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock5 = Multi_Scale_Temporal_Block(outc=2)

        self.fc = nn.Linear(640,256)

    def forward(self,x):
        # [ batch_size, 5, 19, 256]
        t_fe1 = self.mstblock1(x[:,0,:,:].unsqueeze(1))
        t_fe2 = self.mstblock2(x[:,1,:,:].unsqueeze(1))
        t_fe3 = self.mstblock3(x[:,2,:,:].unsqueeze(1))
        t_fe4 = self.mstblock4(x[:,3,:,:].unsqueeze(1))
        t_fe5 = self.mstblock5(x[:,4,:,:].unsqueeze(1))
        t_fe = torch.cat((t_fe1,t_fe2,t_fe3,t_fe4,t_fe5),1)

        # t_fe = self.fc(t_fe)

        return t_fe


# 生成邻接矩阵
class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio=128):
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias=False),
                                nn.ELU(inplace=False),
                                nn.Linear(inc // reduction_ratio, inc, bias=False),
                                nn.Tanh(),
                                nn.ReLU(inplace=False))

    def forward(self, x):
        y = self.fc(x)
        return y


class resGCN(nn.Module):
    def __init__(self, inc, outc,band_num):
        super(resGCN, self).__init__()
        self.GConv1 = nn.Conv2d(in_channels=inc,
                                out_channels=outc,
                                kernel_size=(1, 3),
                                stride=(1, 1),
                                padding=(0, 0),
                                groups=band_num,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.GConv2 = nn.Conv2d(in_channels=outc,
                                out_channels=outc,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 1),
                                groups=band_num,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_p, L):
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        y = torch.einsum('bijk,kp->bijp', (x, L))
        y = self.ELU(torch.add(y, x_p))
        return y



class HGCN(nn.Module):
    def __init__(self, dim, chan_num,band_num):
        super(HGCN, self).__init__()
        self.chan_num = chan_num
        self.dim = dim
        self.resGCN = resGCN(inc=dim * band_num,
                              outc=dim * band_num,band_num=band_num)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x, A_ds):
        x = x.permute(0,1,3,2)
        L = torch.einsum('ik,kp->ip', (A_ds, torch.diag(torch.reciprocal(sum(A_ds)))))
        G = self.resGCN(x, x, L).contiguous()
        return G.squeeze(2).transpose(2,1)

class SGCN(nn.Module):
    def __init__(self,device,chan_num=19,dim=5):
        super().__init__()
        self.chan_num = chan_num
        self.band_num = 5
        self.A = torch.rand((1, self.chan_num * self.chan_num), dtype=torch.float32, requires_grad=False).to(device)
        self.GATENet = GATENet(self.chan_num * self.chan_num, reduction_ratio=chan_num)
        self.gcn = HGCN(dim = dim, chan_num = self.chan_num,band_num=self.band_num)

    def forward(self,x):
        A_ds = self.GATENet(self.A)
        A_ds = A_ds.reshape(self.chan_num, self.chan_num)
        feat = self.gcn(x,A_ds)

        return feat


    