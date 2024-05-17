import argparse
import os
import torch
import pickle

from exp.exp_activity import Exp_Activity
from datautils import *
from sklearn.model_selection import KFold,StratifiedKFold, train_test_split
from torch.utils.data import Subset
import random
import numpy as np

from person_activity import *
from utils.tools import write_file_binary

def training(args):
  
  Exp = Exp_Activity
  
  train_dataset = PersonActivity('/cluster/work/medinfmk/TSDataset/activity/', download=False)
  #test_dataset = PhysioNet('/cluster/work/medinfmk/TSDataset/physionet/', train=True, download=False)
  #train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2)
  
  #train_dataset = Subset(dataset, train_idx)
  #test_dataset = Subset(dataset, test_idx)
  
  setting = '{}_{}'.format(args.model,args.data)
  folder_path = './results/' + setting + '/'+str(args.save_exp)+'/'
  
  if not os.path.exists(folder_path):
      os.makedirs(folder_path)
  
  train_dataset,test_dataset = train_test_split(train_dataset,test_size=0.2,shuffle = True,random_state = args.random_seed)
  
  test_dataset,valid_dataset = train_test_split(test_dataset,shuffle = True,test_size=0.5,random_state = args.random_seed)
  print(len(train_dataset),len(valid_dataset),len(test_dataset))
  if args.is_training:
      for ii in range(args.itr):
          # setting record of experiments
          AUROC,ACC,AUPRC, PRE, REC = [],[],[],[],[]
      
          
          print("----Finish second preprocessing in fold ", ii , " ------------")
          
          
          #train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
          #valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_index)
          #test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
          
          
          
          exp = Exp(args)  # set experiments
          print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
          #exp.train(setting, train_dataset,valid_dataset,test_dataset,ii)
          exp.train(setting, train_dataset, valid_dataset,test_dataset,ii)
  
          print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
          auroc,acc,auprc, precision, recall = exp.test(setting, test_dataset,ii)
          
          AUROC.append(auroc)
          ACC.append(acc)
          AUPRC.append(auprc)
          PRE.append(precision)
          REC.append(recall)
          
          if args.do_predict:
              print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
              exp.predict(setting, test_loader, True)
  
          torch.cuda.empty_cache()
          write_file_binary(folder_path,ii,setting,args,AUROC,ACC,AUPRC, PRE, REC)
          
          
          
  return {"AUC":auroc,"acc":acc,"auprc":auprc,"pre":precision,"recall":recall}
if __name__ == '__main__':
  print("+++++++++++++++++++++++")
  parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
  
  # random seed
  parser.add_argument('--random_seed', type=int, default=0, help='random seed')
  
  # basic config
  parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
  parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
  parser.add_argument('--model', type=str, required=True, default='Autoformer',
                      help='model name, options: [Autoformer, Informer, Transformer]')
  
  # data loader
  parser.add_argument('--data', type=str, default='activity', help='dataset type')
  parser.add_argument('--data_dir', type=str, default='/cluster/work/medinfmk/ICU_Delirium/data/content/datav5/fordm/', help='root path of the data file')
  parser.add_argument('--label_dir', type=str, default='/cluster/work/medinfmk/ICU_Delirium/data/content/datav5/v5_scores.json', help='label file')
  parser.add_argument('--checkpoints', type=str, default='/cluster/work/medinfmk/TSDataset/xingyu_data/checkpoints/', help='location of model checkpoints')
  parser.add_argument('--classify_pertp',action='store_true' )
  parser.add_argument('--save_exp', type=str, default='', help='location of different exp')
  # Experiment
  parser.add_argument('--cv', type=int, default=1, help='Fold of cross validation')
  # PatchTST
  parser.add_argument('--seq_len', type=int, default=2880, help='input sequence length')
  parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
  parser.add_argument('--patch_len', type=int, default=16, help='patch length')
  parser.add_argument('--stride', type=float, default=0.0, help='range of each pooled attention attended to')
  parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
  
  #patchTST
  #num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1 
  #c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
  #parser.add_argument('--channel_in', type=int, default=3, help='input channels')
  parser.add_argument('--class_num', type=int, default=2, help='number of classes')
  
  # Formers 
  parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
  parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
  parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
  parser.add_argument('--c_out', type=int, default=7, help='output size')
  parser.add_argument('--d_model', type=int, default=8, help='dimension of model')
  parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
  parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
  parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
  parser.add_argument('--d_ff', type=int, default=8, help='dimension of fcn')
  parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
  parser.add_argument('--factor', type=int, default=1, help='attn factor')
  parser.add_argument('--distil', action='store_false',
                      help='whether to use distilling in encoder, using this argument means not using distilling',
                      default=True)
  parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
  parser.add_argument('--head_dropout', type=float, default=0.1, help='dropout')
  parser.add_argument('--embed', type=str, default='timeF',
                      help='time features encoding, options:[timeF, fixed, learned]')
  parser.add_argument('--activation', type=str, default='gelu', help='activation')
  parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
  parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
  parser.add_argument('--divide', type=int,help='ratio to reduce patch length')
  parser.add_argument('--grid', type=int,help='length of latent space')
  
  #gru
  parser.add_argument('--gru_out', type=int, default=8, help='dimension of out dimension in gru')
  # optimization
  parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
  parser.add_argument('--itr', type=int, default=2, help='experiments times')
  parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
  parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
  parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
  parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
  parser.add_argument('--des', type=str, default='test', help='exp description')
  parser.add_argument('--loss', type=str, default='mse', help='loss function')
  parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
  parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
  parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
  
  # GPU
  parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
  parser.add_argument('--gpu', type=int, default=0, help='gpu')
  parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
  parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
  parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
  
  args = parser.parse_args()
  
  # random seed
  fix_seed = args.random_seed
  random.seed(fix_seed)
  torch.manual_seed(fix_seed)
  np.random.seed(fix_seed)
  
  #args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
  if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
  else:
      device = torch.device('cpu')
      
  if args.use_gpu and args.use_multi_gpu:
      args.devices = args.devices.replace(' ', '')
      device_ids = args.devices.split(',') 
      args.device_ids = [int(id_) for id_ in device_ids]
      args.gpu = args.device_ids[0]
  
  print('Args in experiment:')
  print(args)
  
  training(args)
  