import os
import datautils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
#from torchvision.datasets.utils import download_url
from utils.tools import locf_torch

def get_data_min_max(records, device):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
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


class PhysioNet(object):

    # urls = [
    #  'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
    #  'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    # ]

    # , '/cluster/work/medinfmk/TSDataset/physionet/set-b.tar.gz']
    ts_files = ['/cluster/work/medinfmk/TSDataset/physionet/set-c.tar.gz']
    outcome_files = [
       '/cluster/work/medinfmk/TSDataset/physionet/Outcomes-c.txt']
    #outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay",
              "Survival", "In-hospital_death"]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root,train=True, download=False,
                 quantization=0.016, n_samples=None, set = False,device=torch.device("cpu")):

        self.root = root
        self.train = train
        self.device = device
        self.reduce = "average"
        self.quantization = quantization
        self.set = set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        if self.train:
            data_file1,data_file2,data_file3 = self.training_file
            label_file1,label_file2,label_file3 = self.training_label
            if self.device == 'cpu':
                self.data = torch.load(os.path.join(self.processed_folder, data_file1), map_location='cpu')+torch.load(os.path.join(self.processed_folder, data_file2), map_location='cpu')+torch.load(os.path.join(self.processed_folder, data_file3), map_location='cpu')
                self.labels = list(torch.load(os.path.join(self.processed_folder, label_file1), map_location='cpu').values())+list(torch.load(os.path.join(self.processed_folder, label_file2), map_location='cpu').values())+list(torch.load(os.path.join(self.processed_folder, label_file3), map_location='cpu').values())
            else:
                self.data = torch.load(os.path.join(self.processed_folder, data_file1))+torch.load(os.path.join(self.processed_folder, data_file2))+torch.load(os.path.join(self.processed_folder, data_file3))
                self.labels = list(torch.load(os.path.join(self.processed_folder, label_file1)).values())+list(torch.load(os.path.join(self.processed_folder, label_file2)).values())+list(torch.load(os.path.join(self.processed_folder, label_file3)).values())
        else:
            data_file = self.test_file
            label_file = self.test_label

            if self.device == 'cpu':
                self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
                self.labels = torch.load(os.path.join(self.processed_folder, label_file), map_location='cpu')
            else:
                self.data = torch.load(os.path.join(self.processed_folder, data_file))
                self.labels = list(torch.load(os.path.join(self.processed_folder, label_file)).values())

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]
        
        # load empirical means
        #self.mean = empirical_mean()
        self.black_list = ['140501', '150649', '140936', '143656', '141264', '145611', '142998', '147514', '142731', '150309', '155655', '156254']
     
    def download(self):
        if self._check_exists():
            return

        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for txtfile in self.outcome_files:
            filename = txtfile.rpartition('/')[2]
            print("loading label files: ", filename)
            #download_url(url, self.raw_folder, filename, None)

            #txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                out_labels = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)
                    out_labels[record_id] = torch.Tensor(labels)[4].to(self.device)
                    
                print("saving label files: ", os.path.join(self.processed_folder,filename.split('.')[0] + '.pt'))
                
                torch.save(
                    out_labels,
                    os.path.join(self.processed_folder,
                                 filename.split('.')[0] +"_" + str(self.quantization)+ '.pt') 
                )

        for url in self.ts_files:
            filename = url.rpartition('/')[2]
            
            #download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(url, "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(self.params)).to(self.device)]
                    mask = [torch.zeros(len(self.params)).to(self.device)]
                    nobs = [torch.zeros(len(self.params))]
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        #print(time)
                        # Time in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        #print("after process: ", time)
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        #time = round(time / self.quantization) * \ 
                        #    self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(
                                len(self.params)).to(self.device))
                            mask.append(torch.zeros(
                                len(self.params)).to(self.device))
                            nobs.append(torch.zeros(
                                len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            #vals[-1][self.params_dict[param]] = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations +
                                           float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        #else:
                            #print("p:",param,record_id,l)
                            #assert param == 'RecordID', 'Read unexpected param {}'.format(param)
                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)
                

                labels = None
                if record_id in outcomes:
                    # Only training set has labels
                    labels = outcomes[record_id]
                    # Out of 5 label types provided for Physionet, take only the last one -- mortality
                    labels = labels[4]
                
                patients.append((record_id, tt, vals, mask, labels))
            print("saving data to ",os.path.join(self.processed_folder,filename.split('.')[0] + "_" + str(self.quantization) + '.pt'))
            torch.save(
                patients,
                os.path.join(self.processed_folder,
                             filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            )

        print('Done!')
 
    def _check_exists(self):
        for url in self.ts_files:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                os.path.join(self.processed_folder,
                             filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        if self.set ==False:
            return 'set-a_{}.pt'.format(self.quantization),'set-b_{}.pt'.format(self.quantization),'set-c_{}.pt'.format(self.quantization)
        else:
            return 'set-a-set_{}.pt'.format(self.quantization),'set-b-set_{}.pt'.format(self.quantization),'set-c-set_{}.pt'.format(self.quantization)

    @property
    def test_file(self):
        return 'set-c_{}.pt'.format(self.quantization)

    #@property
    #def label_file(self):
    #    return 'Outcomes-train.pt'
    @property
    def training_label(self):
        return 'Outcomes-a.pt','Outcomes-b.pt','Outcomes-c.pt'
    @property
    def test_label(self):
        return 'Outcomes-c.pt'

    def __getitem__(self, index):
        return self.data[index]
 
    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(
            'train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str

    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        non_zero_attributes = (torch.sum(mask, 0) > 2).numpy()
        non_zero_idx = [i for i in range(
            len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
        n_non_zero = sum(non_zero_attributes)

        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}

        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(
            n_row, n_col, figsize=(width, height), facecolor='white')

        # for i in range(len(self.params)):
        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:, param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.]
            data_cur_param = data[tp_mask == 1., param_id]

            ax_list[i // n_col, i %
                    n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(),  marker='o')
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)


def variable_time_collate_fn(batch, device=torch.device("cpu"), data_type="train",
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device) # include all possible time steps
    #print("combined tt: ", combined_tt.shape, "inverse indices: ", inverse_indices.shape) 

    offset = 0
    #combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    #combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_vals = torch.zeros([len(batch), 2880, D]).to(device)
    combined_mask = torch.zeros([len(batch), 2880, D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(
        len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        #tt = tt*60/2880.0
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = datautils.normalize_masked_data(combined_vals, combined_mask,
                                                      att_min=data_min, att_max=data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "observed_data": combined_vals,
        "observed_tp": combined_tt,
        "observed_mask": combined_mask,
        "labels": combined_labels}

    #data_dict = utils.split_and_subsample_batch(
    #    data_dict, args, data_type=data_type)
    return data_dict


def variable_time_collate_fn2(batch, args, device=torch.device("cpu"), data_type="train",
                              data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_impute_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        enc_impute_vals[b,:currlen] = locf_torch(vals,mask).to(device)

    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(
        len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = datautils.normalize_masked_data(combined_vals, combined_mask,
                                                      att_min=data_min, att_max=data_max)
    enc_combined_vals, _, _ = datautils.normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                          att_min=data_min, att_max=data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    data_dict = {
        "enc_data": enc_combined_vals,
        "enc_mask": enc_combined_mask,
        "enc_time_steps": enc_combined_tt,
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(
        data_dict, args, data_type=data_type)
    return data_dict


def variable_time_collate_fn3(batch, device=torch.device("cpu"), data_type="train",
                              data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations (may be irregular in different samples).
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]-4
    
    data_min = data_min[4:]
    data_max = data_max[4:]
    
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    #max_len = 2880
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device) # pad time steps and values with zeros
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    #enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.empty([len(batch),maxlen,D]).fill_(-1).to(device)
    
    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(
        len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)
    
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals[:,4:].to(device)
        enc_combined_mask[b, :currlen] = mask[:,4:].to(device)
        if labels is not None:
            combined_labels[b] = labels
    enc_combined_vals, _, _ = datautils.normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                          att_min=data_min, att_max=data_max)

    #if torch.max(enc_combined_tt) != 0.:
    #    enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    data_dict = {
        "observed_data": enc_combined_vals,
        "observed_tp": enc_combined_tt,
        "observed_mask": enc_combined_mask,
        "labels": combined_labels}

    return data_dict
    
    
def variable_time_collate_fn_grud(batch, device=torch.device("cpu"), data_type="train",
                              data_min=None, data_max=None):
    
    D = batch[0][2].shape[1]-4
    
    data_min = data_min[4:]
    data_max = data_max[4:]
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    #max_len = 2880
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device) # pad time steps and values with zeros
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    #enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.empty([len(batch),maxlen,D]).fill_(0).to(device)
    enc_impute_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_impute_time = torch.zeros([len(batch), maxlen, D]).to(device)
    
    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(
        len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)
    
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals[:,4:].to(device) # remove demographic features
        enc_combined_mask[b, :currlen] = mask[:,4:].to(device)
        enc_impute_vals[b,:currlen] = locf_torch(vals,mask)[:,4:].to(device)
        
        tt = tt.unsqueeze(-1).repeat(1,D)
        
        #print("before: ",tt[:10],mask[:10],locf_torch(tt,mask)[:10])
        #print(tt.shape, mask.shape, vals.shape)
        enc_impute_time[b,:currlen] = locf_torch(tt,mask[:,4:])[:,:].to(device)
        
        if labels is not None:
            combined_labels[b] = labels
    enc_combined_vals, _, _ = datautils.normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                          att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    data_dict = {
        "observed_data": enc_combined_vals,
        "observed_tp": enc_combined_tt,
        "observed_mask": enc_combined_mask,
        "observed_impute": enc_impute_vals,
        "impute_tp":enc_impute_time,
        "labels": combined_labels}

    return data_dict


def variable_time_collate_fn_uni(batch, device=torch.device("cpu"), data_type="train",
                              data_min=None, data_max=None):
    
    '''
    Returns:
      combined_tt: The uniform of 48 x 60 minutes, each irregular time step is mapped, 00:00 is removed (index 0 is the 1st minute)
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    '''
    D = batch[0][2].shape[1]-4
    
    data_min = data_min[4:]
    data_max = data_max[4:]
    len_tt = [ex[1].size(0) for ex in batch]
    
    maxlen = 480#2880
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device) # pad time steps and values with zeros
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    #enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.empty([len(batch),maxlen,D]).fill_(0).to(device)
    enc_impute_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    
    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(
        len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)-1 #remove first time step 00:00
        
        #print(max(tt),min(tt))
        tt = tt[1:] -1# back to minutes
        tt = torch.round(tt / 0.1) * 0.1 #round to every 6 min
        #print(max(tt))
        idx = tt.long()
        enc_combined_tt[b,idx] = tt.to(device)
        enc_combined_vals[b, idx] = vals[1:,4:].to(device)
        enc_combined_mask[b, idx] = mask[1:,4:].to(device)
        enc_impute_vals[b,idx] = locf_torch(vals,mask)[1:,4:].to(device)
        
        if labels is not None:
            combined_labels[b] = labels
    enc_combined_vals, _, _ = datautils.normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                          att_min=data_min, att_max=data_max)

    #if torch.max(enc_combined_tt) != 0.:
    #    enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
    
    
    data_dict = {
        "observed_data": enc_combined_vals,
        "observed_tp": enc_combined_tt,
        "observed_mask": enc_combined_mask,
        "observed_impute": enc_impute_vals,
        "labels": combined_labels}

    return data_dict

def variable_time_collate_fn_test(batch, device=torch.device("cpu"), data_type="train",
                              data_min=None, data_max=None, patch = 0):
    
    '''
    Returns:
      combined_tt: The uniform of max batch length, 00:00 is removed (index 0 is the 1st minute)
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    '''
    
    D = batch[0][2].shape[1]
    #D = 2
    
    data_min = data_min
    data_max = data_max
    len_tt = [ex[1].size(0) for ex in batch]
    
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device) # pad time steps and values with zeros
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    #enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.empty([len(batch),maxlen,D]).fill_(0).to(device)
    enc_impute_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    
    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(
        len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)
    count=0
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0) #remove first time step 00:00
        
        #print(max(tt),min(tt))
        #print(tt)
        #tt = tt[1:] * 60 -1# back to minutes
        #print("after: ",tt)
        #tt = tt.squeeze()
        
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
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
        #enc_combined_tt = torch.nan_to_num(enc_combined_tt / torch.nan_to_num(torch.max(enc_combined_tt,dim = 0)[0]))
    
    data_dict = {
    "observed_data": enc_combined_vals,
    "observed_tp": enc_combined_tt,
    "observed_mask": enc_combined_mask,
    #"observed_impute": enc_impute_vals,
    "labels": combined_labels}
    
    return data_dict




if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dataset = PhysioNet(
        '/cluster/work/medinfmk/TSDataset/physionet/', train=True, download=True)
    data_min, data_max = get_data_min_max(dataset, device)

   
