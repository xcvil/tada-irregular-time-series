import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize


    
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def AUROC(pred,true):
    return roc_auc_score(true, pred[:,1])
  
def ACC(pred,true):
    return accuracy_score(true, np.argmax(pred,axis =1))


def AUPRC(y_pred,y_true):
    
    return average_precision_score(y_true,y_pred[:,1])
def Precision(y_pred,y_true):
    return precision_score(y_true,np.argmax(y_pred,axis =1))
    
def Recall(y_pred,y_true):    
    return recall_score(y_true, np.argmax(y_pred,axis =1))



def confusion_matrix_multi(y_pred,y_true,n_classes = 8):
    y_pred_label = np.argmax(y_pred, axis = 1)
    TP,TN,FP,FN = dict(), dict(), dict(),dict()
    precision, recall = dict(), dict()
    for i in range(n_classes):
        temp_true = [1 if p == i else 0 for p in y_true]
        temp_pred = [1 if p == i else 0 for p in y_pred_label]
        
        TP[i] = true_positive(temp_true,temp_pred)
        TN[i] = true_negative(temp_true,temp_pred)
        FP[i] = false_positive(temp_true,temp_pred)
        FN[i] = false_negative(temp_true,temp_pred)
        
        precision[i] = TP[i]/(TP[i]+FP[i]+1e-06)
        recall[i] = TP[i]/(TP[i]+FN[i]+1e-06)
    
    return TP, TN, FP, FN, recall, precision
        
    
def metric(pred, true):
    
    auroc = AUROC(pred,true)
    acc = ACC(pred,true)
    auprc = AUPRC(pred, true)
    precision = Precision(pred, true)
    recall = Recall(pred, true)
    
    return auroc,auprc
    
    
    return auroc,auprc