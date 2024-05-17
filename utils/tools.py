import numpy as np
import torch
#import matplotlib.pyplot as plt
import time

import pickle
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd

#plt.switch_backend('agg')

def write_file(folder_path,fold,setting,auroc,acc,auprc, precision, recall, auprc_p, precision_p, recall_p):
    f = open(folder_path+"_result"+args.d_model+"_seed"+args.random_seed+"_patch"+str(args.patch_len)+"_div"+str(args.divide)+str(args.grid)+"_grid"+".txt", 'w')
    f.write(setting + "  \n")
    f.write('>>>>>>>>>>>>Fold '+str(i+1)+' <<<<<<<<<<<<<<')
    f.write('\n')
    f.write('auroc:{}, acc:{}, auprc:{}, precision:{}, recall:{}'.format(auroc, acc, auprc, precision, recall))
    f.write('\n')
    for i in auprc_p.keys():
        f.write('class :'+str(i)+', auprc:{}, precision:{}, recall:{}'.format(auprc_p[i],precision_p[i],recall_p[i]))
        f.write('\n')
    f.write('\n')
    f.close()
    

def write_file_binary(folder_path,fold,setting,args,auroc,auprc):
    f = open(folder_path+str(args.d_model)+"_seed"+str(args.random_seed)+"_patch"+str(args.patch_len)+"_div"+str(args.divide)+"_grid"+str(args.grid)+"_layer"+str(args.e_layers)+".txt", 'w')
    f.write(setting + "  \n")
    f.write(str(args) + "  \n")
    for i in range(len(auroc)):
        f.write('>>>>>>>>>>>>Fold '+str(i+1)+' <<<<<<<<<<<<<<')
        f.write('\n')
        f.write('auroc:{}, auprc:{}'.format(auroc[i], auprc[i]))
        f.write('\n')
    f.close()

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else max(1e-5,args.learning_rate * (0.9 ** ((epoch - 3) // 1)))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, fold=1):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, fold)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, fold)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, fold):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to '+path + '/cv_' +str(fold)+ '_checkpoint.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def locf_torch(X,mask):
        """Torch implementation of LOCF.
        Parameters
        ----------
        X : tensor,
            Time series containing missing values (NaN) to be imputed.
        Returns
        -------
        X_imputed : tensor,
            Imputed time series.
        """
        
        mask = mask.int()
        idx = torch.cummax(mask, dim=-1)[1]
        
        collector = []
        for v, i in zip(X, idx):
            collector.append(v[i].unsqueeze(0))
        #print(idx.shape,trans_X.shape,max(idx))
        X_imputed = torch.concat(collector, dim=0)
        
        # If there are values still missing,
        # they are missing at the beginning of the time-series sequence.
        # Impute them with self.nan
        if torch.isnan(X_imputed).any():
            X_imputed = torch.nan_to_num(X_imputed, nan=0)

        return X_imputed

def load_mimic_iii(file):
    data, oc, train_ind, valid_ind, test_ind = torch.load(file)
    data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
    
    
    # Fix age.
    data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
    # Get static data with mean fill and missingness indicator.
    static_varis = ['Age', 'gender']
    ii = data.variable.isin(static_varis)
    static_data = data.loc[ii]
    data = data.loc[~ii]
    def inv_list(l, start=0):
        d = {}
        for i in range(len(l)):
            d[l[i]] = i+start
        return d
    static_var_to_ind = inv_list(static_varis)
    D = len(static_varis)
    N = data.ts_ind.max()+1
    demo = np.zeros((N, D))
    for row in tqdm(static_data.itertuples()):
        demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
    