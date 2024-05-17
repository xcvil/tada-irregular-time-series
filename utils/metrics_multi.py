import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn
    
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

def AUROC_multi(pred,true):
    print(true.shape, pred.shape)
    return roc_auc_score(true, pred, average = 'macro',multi_class = 'ovr')

def AUROC(pred,true):
    return roc_auc_score(true, pred[:,1])
  
def ACC(pred,true):
    return accuracy_score(true, np.argmax(pred,axis =1))
def ACC_multi(y_pred,y_true):
    
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    y_pred = np.argmax(y_pred,axis =1)
    for yt, yp in zip(y_true, y_pred):
        
        if yt == yp:
            
            correct_predictions += 1
    
    #returns accuracy
    return correct_predictions / len(y_true)

def AUPRC_multi(y_pred,y_true,n_classes = 8):
    
    y_true = label_binarize(y_true, classes=[*range(n_classes)])
    '''
    average_precision = []
    precision, recall = dict(), dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:,i],
                                                            y_pred[:, i])
        #print("label ", i," precision and recall score: ", precision[i], recall[i])
        average_precision.append(average_precision_score(y_true[:, i], y_pred[:, i]))
        
    
    return sum(average_precision)/len(average_precision)
    '''
    return average_precision_score(y_true,y_pred,average="macro")
def Precision_multi(y_pred,y_true):
    return precision_score(y_true,np.argmax(y_pred,axis =1),average = 'macro')
    
def Recall_multi(y_pred,y_true):    
    return recall_score(y_true, np.argmax(y_pred,axis =1),average = 'macro')


def AUPRC_per_class(y_pred,y_true,n_classes = 8):
    
    y_true_bi = label_binarize(y_true, classes=[*range(n_classes)])
    y_pred_label = np.argmax(y_pred, axis = 1)
    
    unique, counts = np.unique(y_pred_label, return_counts=True)
    #print("predicted: ",dict(zip(unique, counts)))
    unique, counts = np.unique(y_true, return_counts=True)
    #print(" treu: , " ,dict(zip(unique, counts)))
    average_precision = dict()
    precision, recall = dict(), dict()
    for i in range(n_classes):
        temp_true = [1 if p == i else 0 for p in y_true]
        temp_pred = [1 if p == i else 0 for p in y_pred_label]
        
        precision[i] = precision_score(temp_true,temp_pred)
        recall[i] = recall_score(temp_true,temp_pred)
        #precision[i], recall[i], _ = precision_recall_curve(y_true[:,i],
        #                                                    y_pred[:, i])
        #print("label ", i," precision and recall score: ", precision[i], recall[i])
        average_precision[i]= average_precision_score(y_true_bi[:, i], y_pred[:, i])
        
    
    return average_precision, precision, recall
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
        
    
def metric_multi(pred, true,n_classes = 7):
    #print(" in cal metric: ", pred.shape, true.shape, np.max(pred,axis = 1).shape)
    #print(pred[:5],true[:5])
    pred_scores = pred
    idx = np.argmax(pred_scores, axis=-1)
    preds_label = np.zeros(pred_scores.shape)
    preds_label[np.arange(preds_label.shape[0]), idx] = 1
    
    acc = ACC_multi(pred,true)
    auprc = AUPRC_multi(pred, true,n_classes)
    precision = Precision_multi(pred, true)
    recall = Recall_multi(pred, true)
    
    return auprc,acc,auprc, precision, recall
    
    
    return acc,auprc