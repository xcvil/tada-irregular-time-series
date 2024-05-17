from models.tada_layers.modules import *
import os
import einops
from torch.autograd import Variable
import numpy as np
import math
from typing import Any
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F
from torch import Tensor





class ClassificationHead(nn.Module):
    def __init__(self,fea, n_classes, head_dropout,classify_pertp= False): 
        super().__init__()    
        
        self.dropout = nn.Dropout(head_dropout)
        
        if classify_pertp==False:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Linear(fea,n_classes)
        else:
            self.pool = None
            self.flatten = nn.Linear(fea,n_classes)
        

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        
        if self.pool!=None:
            x = self.pool(x.transpose(2,1)).squeeze(-1)       # y: bs x n_classes
        x = self.flatten(x)
       
        return x        


class TimeEmbedding(nn.Module):
    def __init__(self, configs,input_size,hidden_size,n_heads):
        
        super(TimeEmbedding, self).__init__()
        
        self.sdim = 2
        self.d_model = hidden_size
        d_model = hidden_size//n_heads
        
        
        self.K=nn.Linear(d_model+self.sdim,d_model)
        
        self.h = nn.Sequential(nn.Linear(self.sdim,d_model),nn.ReLU(),nn.Linear(d_model, d_model))
        self.g = nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU(),nn.Linear(d_model, d_model))
        
        self.Q = nn.Parameter(torch.zeros(n_heads,d_model))
        self.mlp = nn.Sequential(
            nn.Linear(self.sdim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        
        self.apply(self.init_weights)
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def forward(self,x,index,mask=None):
        # x b l d 1
        batch = x.shape[0]
        length = x.shape[1]
        dim = x.shape[2]
        index = index.unsqueeze(-1)
        
        x = x.unsqueeze(-1)
        
        s = torch.cat([x,index],-1)
        
        if mask!=None:
            fs = self.mlp(s).sum(-2)
            count = mask.sum(-1).unsqueeze(-1)
            fs = fs/(count+1e-9)
        else:
            fs = self.mlp(s).mean(-2)
        
        fs = self.g(fs)
        fs = fs.unsqueeze(-2).repeat(1,1,dim,1)
        fs = torch.cat([fs,s],-1)
        Z = self.K(fs)
        e = torch.einsum('blsd,md->blsm', Z, self.Q)
        if mask is not None:
            e = e.masked_fill(mask.unsqueeze(-1) == 0, -1e9 )
        
        e = torch.softmax(e,dim = -2)
        hs = self.h(s)
        r =  torch.einsum('blsd,blsm->blmd', hs, e)
        r = r.reshape(batch,-1,self.d_model) 
        return r 


class Model(nn.Module):
    def __init__(self, configs,device,proj_dim = 16,num_inds=32,pos_dim = 3,ln=True):
        super(Model, self).__init__()
        
        dim_input = configs.enc_in*2+1#pos_dim
        dim_hidden = configs.d_model
        num_heads = configs.n_heads
        divide = configs.divide
        self.grid = configs.grid
        self.stride = configs.stride
        self.patch = configs.patch_len
        
        
        self.dim_enc = TimeEmbedding(configs,2,dim_hidden,1)
        
        self.stride_list = torch.zeros(configs.enc_in*2+1).to(device)
        
        self.att = DLA(configs.enc_in*2+1, dim_hidden, dim_hidden+1, num_heads)
        
        self.patch_embed = nn.Linear(self.patch,self.patch)
        self.act = nn.ReLU()
        self.enc = nn.ModuleList()
        
        #print("config e layer: ",configs.e_layers)
        for i in range(1,configs.e_layers+1):
            print((math.ceil((self.grid+1)/self.patch))/divide**(i-1))
            self.enc.append(scale_block(dim_hidden,(math.ceil((math.ceil((self.grid+1)/self.patch))/divide**(i-1))), num_heads, self.patch,divide,configs.dropout))
            
        
                
                
        if configs.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                configs.gpu) if not configs.use_multi_gpu else configs.devices
            self.device = torch.device('cuda:{}'.format(configs.gpu))
            
        else:
            self.device = torch.device('cpu')
            
        self.query =  torch.nn.Parameter(torch.zeros(self.grid,dim_hidden+1)).to(self.device)
        self.dec_pos_embedding = nn.Parameter(torch.zeros(1, 1, self.patch, dim_hidden))   
        self.fuse =  nn.Sequential(
            nn.Linear(dim_hidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU())
        self.classifier = ClassificationHead(300, configs.class_num, configs.dropout,configs.classify_pertp)
        self.avgpool = nn.AdaptiveAvgPool1d(int(self.grid))
        if configs.classify_pertp:
            self.dec = nn.Linear(int(self.grid),50)
        else:
            self.dec = None
    def forward(self, X,mean=None):
        
        timesteps = X['observed_tp'].float().to(self.device)
        batch_x = X['observed_data'].float().to(self.device)
        mask = X['observed_mask'].float().to(self.device)
        
        batch_num = batch_x.shape[0]
        
        
        z = torch.cat((batch_x, mask), 2)
        mask = torch.cat((mask, mask), 2)
        #z = batch_x
        q = self.query.unsqueeze(0).repeat(batch_num,1,1)
        
        ref = torch.linspace(0, 1., self.grid).to(self.device)
        
        t = timesteps.unsqueeze(-1)
        
        v = torch.cat([z,timesteps.unsqueeze(-1)],-1)
        
        index = torch.arange(batch_x.shape[-1]).unsqueeze(0).unsqueeze(1).repeat(batch_num,batch_x.shape[1],2).to(z.device)
        z = self.dim_enc(z,index,mask)
        
        k = torch.cat([z,t],-1)
        
        
        mask =torch.cat([mask,torch.ones(mask.shape[0],mask.shape[1],1).to(self.device)],-1)
        
        
        z, stride= self.att(q, k, v, mask,ref,timesteps,self.stride_list)
        self.stride_list = stride
        
        patch_num = z.shape[1]//self.patch+1 
        pad = (self.patch*patch_num)-z.shape[1]
        pad = torch.zeros(z.shape[0],pad, z.shape[2]).to(self.device)
        z = torch.cat([z, pad],1) 
        
        z = einops.rearrange(z, "b (p pl) d -> (b p d) pl", pl=self.patch)
        
        z = self.patch_embed(z)
        z = einops.rearrange(z, "(b p d) pl -> b p pl d", pl=self.patch,b = batch_num,p = patch_num)
        
        encode_z = []
        merge_z = None
        out = None
        for block in self.enc:
            if merge_z is None:
                merge_z = z
            
            z,merge_z = block((z,merge_z))
            z = einops.rearrange(z, "b p pl d -> b (p pl) d", pl=self.patch) 
            #print(z.shape)
            z= self.avgpool(z.transpose(2,1)).transpose(2,1)
            if out is None:
                out = z
            else:
                out = z*out
                
            encode_z.append(z) 
        out = self.fuse(out)
        
        if self.dec is not None:
            out = self.dec(out.transpose(2,1)).transpose(2,1)
        
        z = self.classifier(out)
        
        return z               
        


class DLA(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1,stride = None):
        super(DLA, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
                                      #nn.Linear(embed_time, nhidden)])
        self.relu = torch.nn.Sigmoid()
        self.range = nn.Linear(input_dim,input_dim)
        #self.range = nn.Linear(embed_time,embed_time)
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9 )
        
        p_attn = F.softmax(scores, dim = -2)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, qt = None, tt = None,stride = None,dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        ref = qt.shape[0]
        dim = value.shape[-1]
        batch, seq_len, dim = value.size()
        
        value = value.unsqueeze(1)
        
                      
        query = self.linears[0](query).view(query.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
        key = self.linears[1](key).view(key.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
       
        
        tt = tt.unsqueeze(1).unsqueeze(-1).repeat(1,1,1,dim)
        qt = qt.unsqueeze(-1).repeat(1,dim)
        
        all_mask = []
        results = []
        
        stride = self.relu(self.range(stride))
        
        for i in range(0,ref):
            m = (tt<=qt[i]+stride) * (tt>=qt[i]-stride)*mask
            
            all_mask.append(m)
            
        all_mask = torch.stack(all_mask,1).squeeze(2)
                
            
        x, _ = self.attention(query, key, value, all_mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x),stride
        
        return self.linears[-1](x),stride