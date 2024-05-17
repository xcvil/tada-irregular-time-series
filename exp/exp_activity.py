import sys
sys.path.insert(0, '..')
from datautils import *
from person_activity import *
from exp.exp_basic import Exp_Basic
from models import TADA
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, write_file_binary
from utils.metrics_multi import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold



warnings.filterwarnings('ignore')
def empirical_mean(data):
    total,count = 0,0
    for sample in data:
        val = sample[2]
        total+=torch.sum(val, dim = 0)
        count +=val.shape[0]
    return total/count   
class Exp_Activity(Exp_Basic):
    def __init__(self, args):
        super(Exp_Activity, self).__init__(args)
        
        
        self.softmax = nn.Softmax(dim=1)
    def _build_model(self):
        model_dict = {
            'TADA':TADA
        }
        #print(self.args)
        model = model_dict[self.args.model].Model(self.args,device = self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data_loader(self,dataset,subsampler = None,shuffle = False):   
        
        batch_size = self.args.batch_size
        if 'seft' in self.args.model:
            data_min, data_max = None,None
        elif 'hier' in self.args.model or 'raindrop' in self.args.model or 'warp' in self.args.model:
            data_min,data_max = None,None
        else:
            #if os.path.isfile('/cluster/home/chenxin/clinical_ts/dataset/'+str(self.args.random_seed)+'mimic_index.pt'):
            #data_min, data_max = get_data_min_max(dataset, self.device)
            data_min,data_max = None,None
        print(self.args.model)
        print(len(dataset),subsampler)
        if 'GRU' in self.args.model:
            print("1")
            data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: variable_time_collate_fn_grud(batch, self.device, data_type="train", data_min=data_min, data_max=data_max),sampler=subsampler)
        elif 'seft' in self.args.model:
            print("2")
            data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: variable_time_collate_fn_set(batch, self.device, data_type="train", data_min=data_min, data_max=data_max),sampler=subsampler)
        elif '43' in self.args.model:
            data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: variable_time_collate_fn_set(batch, self.device, data_type="train", data_min=data_min, data_max=data_max),sampler=subsampler)
        elif 'Base' in self.args.model or 'Cross' in self.args.model:
            print("3")
            data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: variable_time_collate_fn_activity(batch, self.device, data_type="train", data_min=data_min, data_max=data_max),sampler=subsampler)
            
        else:
            print("4")
            data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: variable_time_collate_fn_activity(batch, self.device, data_type="train", data_min=data_min, data_max=data_max),sampler=subsampler)
        
        mean = empirical_mean(dataset)
        #print("calculate mean: ", mean)
        return data_loader,mean

    def _select_optimizer(self):
    
        all_parameters = set(self.model.parameters())
        range_paras = set(self.model.att.range.parameters())
        comm_layers_params = all_parameters - range_paras
        model_optim = optim.Adam([
                {'params': list(comm_layers_params)},
                {'params': list(range_paras), 'lr': 1e-3}
            ], lr=self.args.learning_rate)
    
        #model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_loader, valid_mean, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, item in enumerate(vali_loader):
                #batch_x = item['observed_data'].float().to(self.device)
                #batch_m = batch_m.float().to(self.device)
                batch_y = item['labels'].to(self.device).squeeze(-1).type(torch.LongTensor)

                # decoder input
                #dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(item,valid_mean)
                        
                else:
                    #if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(item,valid_mean)
                if self.args.classify_pertp:
                    N = batch_y.size(-1)
                    outputs = outputs.reshape(-1, N)
                    batch_y = batch_y.reshape(-1, N)
                    _, batch_y = batch_y.max(-1)
                    
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                
                preds.append(self.softmax(pred))
                trues.append(true)
                
                loss = criterion(pred, true)

                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        
        preds = torch.cat(preds,dim = 0)
        trues= torch.cat(trues,dim =0)
        #print(preds.shape,trues.shape)
        preds = preds.detach().cpu().numpy()#.reshape(-1,2)
        trues = trues.detach().cpu().numpy()
            
        
        auroc,acc,auprc, precision, recall = metric_multi(preds, trues)
        
        print("---------------------------------------------------")
        print("valid accuracy: ", acc,"valid AUROC: ", auroc,"valid AUPRC: ", auprc,"valid precision: ", precision,"valid recall: ", recall)
        print("---------------------------------------------------")
        self.model.train()
        return auroc

    def train(self, setting, train_data, valid_data, test_data,fold):
        #TODO: re-define train test split
        #print("???????????????????????????????????????")
        #train_dataset, train_loader,val_dataset,val_loader,test_dataset,test_loader = self._get_data()
        
        #self.test_dataset = test_dataset
        #self.test_loader = test_loader
        
        train_loader,train_mean = self._get_data_loader(train_data,None,True)
        valid_loader,valid_mean = self._get_data_loader(valid_data,None,False)
        test_loader,test_mean = self._get_data_loader(test_data, None,False)

        path = os.path.join(self.args.checkpoints, setting,self.args.save_exp)
        model_path = path +'/'+ str(self.args.d_model)+"_seed"+str(self.args.random_seed)+"_patch"+str(self.args.patch_len)+"_div"+str(self.args.divide)+"_grid"+str(self.args.grid)+"_layer"+str(self.args.e_layers)+ '_checkpoint.pth'
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            
            preds = []
            trues = []
            
            self.model.train()
            epoch_time = time.time()
            for i, item in enumerate(train_loader):
                #print(item['observed_tp'][0])
                #batch_x = item['observed_data'].float().to(self.device)
                #batch_m = batch_m.float().to(self.device)
                batch_y = item['labels'].squeeze(-1).type(torch.LongTensor).to(self.device)
                
                
                iter_count += 1
                model_optim.zero_grad()
                
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(item,train_mean)
                        
                        loss = criterion(outputs, batch_y)
                        #loss = multi_weighted_logloss(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    #if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(item,train_mean)
                    if self.args.classify_pertp:
                        N = batch_y.size(-1)
                        outputs = outputs.reshape(-1, N)
                        batch_y = batch_y.reshape(-1, N)
                        _, batch_y = batch_y.max(-1)
                    #print(outputs.shape,batch_y.shape)
                    #print("output:",outputs.shape,type(outputs),type(batch_y),batch_y.shape,outputs.get_device(),batch_y.get_device())
                    loss = criterion(outputs, batch_y)
                    #loss = multi_weighted_logloss(outputs, batch_y)
                    train_loss.append(loss.item())
                
                
                pred = self.softmax(outputs) # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            #print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            #print(self.model.stride_list)
            train_loss = np.average(train_loss)
            vali_auroc = self.vali(valid_loader, valid_mean,criterion)
            test_auroc = self.vali(test_loader, test_mean,criterion)
            
            preds = torch.cat(preds,dim = 0)
            trues= torch.cat(trues,dim =0)
            #print(preds.shape,trues.shape)
            preds = preds.detach().cpu().numpy()#.reshape(-1,2)
            trues = trues.detach().cpu().numpy()#.reshape(-1,1)
            #auroc,acc,auprc, precision, recall = metric(preds, trues)
            auroc,acc,auprc, precision, recall = metric_multi(preds, trues)
            print("")
            print("---------------------------------------------------")
            print("training accuracy: ", acc)
            print("training AUROC: ", auroc)
            print("training AUPRC: ", auprc)
            print("precision: ", precision)
            print("recall: ", recall)
            print("---------------------------------------------------")
            print("")
            
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali auroc: {3:.7f},Test auroc: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_auroc,test_auroc))
            early_stopping(-vali_auroc, self.model, model_path, fold)
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args) 
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = model_path
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test_data,fold,test_subsampler = None,test=0):
        
        test_loader,test_mean = self._get_data_loader(test_data,test_subsampler,False)
        path = os.path.join(self.args.checkpoints,setting,self.args.save_exp,str(self.args.d_model)+"_seed"+str(self.args.random_seed)+"_patch"+str(self.args.patch_len)+"_div"+str(self.args.divide)+"_grid"+str(self.args.grid)+"_layer"+str(self.args.e_layers), '_checkpoint.pth')
        
        if test:
            print('loading model from'+path )
            self.model.load_state_dict(torch.load(path))

        preds = []
        trues = []
        #preds = np.array([]).reshape(0,2)
        #trues = np.array([]).reshape(0,1)
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        
        with torch.no_grad():
            for i, item in enumerate(test_loader):
                #batch_x = item['observed_data'].float().to(self.device)
                batch_y = item['labels'].to(self.device).squeeze(-1).type(torch.LongTensor)


               
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(item,test_mean)
                        
                else:
                    #if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(item,test_mean)
                if self.args.classify_pertp:
                    N = batch_y.size(-1)
                    outputs = outputs.reshape(-1, N)
                    batch_y = batch_y.reshape(-1, N)
                    _, batch_y = batch_y.max(-1)    
                outputs = outputs#.detach().cpu().numpy()
                batch_y = batch_y#.detach().cpu().numpy()

                pred = self.softmax(outputs) # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                preds.append(pred)
                trues.append(true)
                #np.vstack([preds,pred])
                #np.vstack([trues,true])
                #inputx.append(batch_x.detach().cpu().numpy())
                
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = torch.cat(preds,dim = 0)
        trues= torch.cat(trues,dim =0)
        #print(preds.shape,trues.shape)
        preds = preds.detach().cpu().numpy()#.reshape(-1,2)
        trues = trues.detach().cpu().numpy()#.reshape(-1,1)
        
        #print(trues)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        #print("check results: ", preds[:2],trues[:2])
        auroc,acc,auprc, precision, recall = metric_multi(preds, trues)
        print("")
        print("-----------------TEST RESULTS----------------------------------")
        print("test accuracy: ", acc)
        print("test AUROC: ", auroc)
        print("test AUPRC: ", auprc)
        print("test precision: ", precision)
        print("test recall: ", recall)
        print("---------------------------------------------------")
        print("")
        

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path +str(fold)+'_pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return auroc,acc,auprc, precision, recall

    def predict(self, pred_loader,setting, load=False):
        #pred_data, pred_loader = self._get_data(flag='pred')
        #_,pred_loader = self._get_data_loader(data, label)

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path +str(fold)+'_real_prediction.npy', preds)

        return
    