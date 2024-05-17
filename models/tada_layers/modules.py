import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent patches will be merged into one patch to
    get representation of a coarser scale
    '''
    def __init__(self, d_model, patch_size,divide, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.divide = divide
        self.patch_size= patch_size//divide
        

    def forward(self, x):
        
        batch_size, p, pl, d_model = x.shape
        if p%self.divide!=0:
            pad_num = p// self.divide +1
            pad_num = pad_num*self.divide - p
        else:
            pad_num = 0
        
        if pad_num != 0: 
            
            x = torch.cat((x, torch.zeros(x.shape[0],pad_num,x.shape[2],x.shape[3]).to(x.device)), dim = 1)
        ts_d = x.shape[2]
        
        x = x.reshape(x.shape[0],x.shape[1]//self.divide,x.shape[2]*self.divide,-1)
        
        return x 
class scale_block(nn.Module):
    
    def __init__(self, dim_hidden, input_len,num_heads,patch_size, divide,dropout,ln = False):
        super(scale_block, self).__init__()

        if (patch_size > 1):
            self.merge_layer = SegMerging(dim_hidden, patch_size,divide, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        
        self.patch =patch_size
        self.divide = divide
        self.encode_layers1=patchSAB(dim_hidden,dim_hidden, input_len,num_heads, ln=ln)
        self.encode_layers2=nn.Linear(patch_size,max(1,patch_size//divide))
        
    
    def forward(self, x):
        x,merge_x = x[0],x[1]
        
        batch = x.shape[0]
        dim = x.shape[-1]
        
        #print('input', merge_x.shape)
        x = self.encode_layers1(merge_x)
        #print('after layers:',x.shape)
        
        x = einops.rearrange(x, "b seg_num ts_d d_model -> (b seg_num d_model) ts_d",b = batch,d_model = dim)
        merge_x = self.encode_layers2(x)
        merge_x = einops.rearrange(merge_x, "(b seg_num d_model) ts_d -> b seg_num ts_d d_model",b = batch,d_model = dim)
        x = einops.rearrange(x, "(b seg_num d_model) ts_d -> b seg_num ts_d d_model",b = batch,d_model = dim)
        
        if self.merge_layer is not None:
            merge_x = self.merge_layer(merge_x) 
        
        return (x,merge_x)
        

        
        
class patchMAB(nn.Module):
    def __init__(self, dim_Q, dim_V, len_in, num_heads, ln=False):
        super(patchMAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_patch = nn.Linear(len_in, len_in)
       
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q):
        
        batch = Q.shape[0]
        Q_r = einops.rearrange(Q, 'b seg_num ts_d d_model -> (b ts_d) seg_num d_model')
        
        Q = self.fc_q(Q_r)
        
        Q = einops.rearrange(Q, 'b seg_num d_model -> b d_model seg_num')
        
        O = self.fc_patch(Q).transpose(-2,-1)
        
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = Q_r + F.relu(self.fc_o(O))
        
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        
        O = einops.rearrange(O, '(b ts_d) seg_num d_model -> b seg_num ts_d d_model', b = batch)
        
        return O

class patchSAB(nn.Module):
    def __init__(self, dim_in, dim_out, input_len,num_heads, ln=False):
        super(patchSAB, self).__init__()
        print('input lengthe:', input_len)
        self.mab = patchMAB(dim_in, dim_out, input_len, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X)
class intrapatchSAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(intrapatchSAB, self).__init__()
        self.mab = intrapatchMAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

