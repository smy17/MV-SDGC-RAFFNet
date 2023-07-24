import torch
import torch.nn as nn
from blocks import *

class Model(nn.Module):
  def __init__(self, device, dataset):
    super().__init__() 
    self.dataset = dataset
    if dataset == 'LANZHOU':
      chan_num=128
    elif dataset == 'HUSM':
      chan_num = 19
    self.device = device
    self.t_block = Temporal_Block()
    self.tgcn = SGCN(device,dim=5,chan_num=chan_num)
    self.dgcn = SGCN(device,dim=1,chan_num=chan_num)
    self.pgcn = SGCN(device,dim=1,chan_num=chan_num)
    if dataset == 'HUSM':
      inc = 512
    else:
      inc = 122
    self.chanattn = nn.Conv2d(in_channels=inc*1,out_channels=32,kernel_size=1)

    self.encoder = RegionAttn(device,n_heads=3,d_model=128,d_k=16,d_v=16,d_ff=32)
    self.cls_token = nn.Parameter(torch.zeros(1,1,128)).to(device)
    self.proj1 = nn.Linear(32*25,64)
    self.proj2 = nn.Linear(5,32)
    self.proj3 = nn.Linear(5,32)
    self.dpout = nn.Dropout(p=0.5)
    self.fc = nn.Linear(64,2)
    self.bn = nn.BatchNorm1d(128)
    self.linear = nn.Linear(128,64)

  def forward(self, filtered, prefeat):
    # [ batch_size, 5, 19, 256]
    temporal_fe = self.t_block(filtered)
    t_fe = self.tgcn(temporal_fe)
    t_fe = self.chanattn(t_fe).permute(0,3,2,1).contiguous()
    t_fe = self.proj1(t_fe.view(prefeat.shape[0],prefeat.shape[1],-1))
    d_fe = self.dgcn(self.rasmGenerate(prefeat[:,:,:5],self.dataset).permute(0,2,1).unsqueeze(3))
    p_fe = self.pgcn(prefeat[:,:,5:].permute(0,2,1).unsqueeze(3))
    d_fe = self.proj2(d_fe)
    p_fe = self.proj3(p_fe)
    fe = torch.cat((t_fe,d_fe,p_fe),dim=2)
    #fe = self.dpout(fe)

    cls_tokens = self.cls_token.expand(filtered.shape[0],-1,-1)
    fe = torch.cat((cls_tokens,fe),dim=1)
    # print(fe.shape)
    
    fe = self.bn(self.encoder(fe)[:,0,:])
    fe = self.linear(fe)
    out = self.fc(fe)
    
    return fe, out

  def rasmGenerate(self,de, dataset='HUSM'):
    # [ batch_size, 19, 5]
    # Fp1, F3, C3, P3, O1, F7, T3, T5, Fz, Fp2, F4, C4, P4, O2, F8, T4, T6, Cz, Pz
    # 0-9, 1-10, 2-11, 3-12, 4-13, 5-14, 6-15, 7-16, 8, 17, 18
    rasm = torch.zeros_like(de)
    if dataset == 'HUSM':
      for chan in range(8):
        rasm[:, chan, :] = de[:, chan, :]/de[:, chan+9, :]
        rasm[:, chan+9, :] = de[:, chan+9, :]/de[:, chan, :]
    elif dataset == 'LANZHOU':
      EEGpairs = [0, 32, 26, 23, 19, 12, 0, 106, 25, 22, 18, 0, 5, 112, 21, 0, 0, 0, 10, 4, 118,
                  14, 9, 3, 124, 8, 2, 123, 117, 111, 105, 80, 1, 122, 116, 110, 104, 87, 121, 115, 109,
                  103, 93, 120, 114, 108, 102, 98, 119, 113, 101, 97, 92, 86, 79, 0, 107, 100, 96, 91, 85,
                  78, 0, 99, 95, 90, 84, 77, 94, 89, 83, 76, 0, 88, 82, 0, 71, 67, 61, 54, 31,
                  0, 74, 70, 66, 60, 53, 37, 73, 69, 65, 59, 52, 42, 68, 64, 58, 51, 47, 63, 57, 
                  50, 46, 41, 36, 30, 7, 56, 45, 40, 35, 29, 13, 49, 44, 39, 34, 28, 20, 48, 43,
                  38, 33, 27, 24, 128, 127, 126, 125]
      for chan in range(128):
        if EEGpairs[chan+1] == 0:
          continue
        else:
          rasm[:, chan, :] = de[:, EEGpairs[chan+1]-1, :]/de[:, chan, :]
        
    return rasm
  



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x1 = torch.rand(64,5,19,256*4).to(device)
x2 = torch.rand(64,19,10).to(device)
model = Model(device,'HUSM').to(device)
_,y = model(x1,x2)
print(y.shape)