import os
import math
import datautils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
import pandas as pd
import warnings

from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from utils.tools import locf_torch

warnings.simplefilter(action='ignore')
'''
def get_data_min_max(records, device):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    for b, (record_id, tt,vals,mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            #print("before: ", vals[:,i])
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            #print(non_missing_vals)
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
                #batch_min.append(torch.tensor(0.0).to(device))
                #batch_max.append(torch.tensor(0.0).to(device))
            else:
                batch_min.append(torch.min(non_missing_vals).to(device))
                batch_max.append(torch.max(non_missing_vals).to(device))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max
'''
def get_data_min_max(records, device):
    all_min = []
    all_max = []
    for k in range(12):
        data_min, data_max = torch.tensor(float('inf')), 0.
        for i, (record_id, tt,vals,mask, labels) in enumerate(records):
            for j in range(vals.shape[0]):
                if mask[j, k]:
                    data_min = min(data_min, vals[j, k])
                    data_max = max(data_max, vals[j, k])
        
        if data_max == 0:
            data_max = torch.tensor(1.0)
        if data_min ==torch.tensor(float('inf')):
            data_min = torch.tensor(0.0)
        all_min.append(data_min)
        all_max.append(data_max)
        #print(data_min, data_max)
    print(all_min,all_max)
    all_min, all_max = torch.stack(all_min).to(device),torch.stack(all_max).to(device)
    return all_min, all_max
        
        
        
class MIMICIV(object):
    
    
    def __init__(self, data, labels, train=True, selected_length = True, n_samples=None, device=torch.device("cpu")):

        #self.root = root
        self.train = train
        self.device = device
        
        

        if not self._check_exists("/cluster/work/medinfmk/TSDataset/xingyu_data/mimiciv/labels.pt"):
            raise RuntimeError('Dataset not found.')

        self.label_file = "/cluster/work/medinfmk/TSDataset/xingyu_data/mimiciv/labels.pt"
    
        self.data_file = "/cluster/work/medinfmk/TSDataset/xingyu_data/mimiciv/data_min_11.pt"
        '''
        if self.device == 'cpu':
            self.data = torch.load(self.data_file)   
            self.labels = torch.load(self.label_file)
        else:
            self.data = torch.load(self.data_file)
            self.labels = torch.load(self.label_file)
        '''
        self.data = data
        self.labels = labels
        
        #print(type(self.data))
        #self.labels = torch.tensor(list(self.labels.values()))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]
        
        #self.processed_folder = 
    def _check_exists(self,path):
        if not os.path.exists(path):
            return False
        return True
    
        
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]
        

def variable_time_collate_fn_test(batch, device=torch.device("cpu"), data_type="train",
                              data_min=None, data_max=None, patch = 0):
    
    '''
    Returns:
      combined_tt: The uniform of max batch length, 00:00 is removed (index 0 is the 1st minute)
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    '''
    
    D = batch[0][2].shape[1]
    #D = 5
    
    data_min = data_min
    data_max = data_max
    len_tt = [ex[1].size(0) for ex in batch]
    
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device) # pad time steps and values with zeros
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    #enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.empty([len(batch),maxlen,D]).fill_(0).to(device)
    enc_impute_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    all_record_ids = []
    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(
        len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)
    count=0
    #black_list = ['140501', '150649', '140936', '143656', '141264', '145611', '142998', '147514', '142731', '150309', '155655', '156254']
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0) #remove first time step 00:00
        
        #print(max(tt),min(tt))
        #print(tt)
        #tt = tt[1:] * 60 -1# back to minutes
        #print("after: ",tt)
        #tt = tt.squeeze()
        all_record_ids.append(record_id)
        idx = tt.long()
        enc_combined_tt[b,:currlen] = tt[:].to(device)
        enc_combined_vals[b, :currlen] = vals[:,:].to(device)
        enc_combined_mask[b, :currlen] = mask[:,:].to(device)
        #enc_impute_vals[b,:currlen] = locf_torch(vals,mask)[:,:].to(device)
        
        if labels is not None:
            combined_labels[b] = labels
    if data_min!=None:
        enc_combined_vals, _, _ = datautils.normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                              att_min=data_min, att_max=data_max)
    #print("test")
    if torch.max(enc_combined_tt) != 0.:
        #print(torch.max(enc_combined_tt,dim = 0))
        #enc_combined_tt = torch.nan_to_num(enc_combined_tt / torch.nan_to_num(torch.max(enc_combined_tt,dim = 0)[0]))
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
        
    data_dict = {
    "observed_data": enc_combined_vals,
    "observed_tp": enc_combined_tt,
    "observed_mask": enc_combined_mask,
    #"observed_impute": enc_impute_vals,
    "labels": combined_labels,
    'record_id':all_record_ids}
    
    return data_dict

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dataset = MIMICIV('/cluster/work/medinfmk/TSDataset/xingyu_data/mimiciv/data/test_csv/', train=True)
    #data_min, data_max = get_data_min_max(dataset, device)
    
    data_min, data_max = get_data_min_max(dataset, device)
    print(data_min,data_max,data_min.shape,data_max.shape)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True,collate_fn=lambda batch: variable_time_collate_fn(
        batch, device = device, data_type="train", data_min=data_min, data_max=data_max))
    for item in dataloader.__iter__().next().values():
        print(item.shape)