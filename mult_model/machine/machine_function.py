#函数
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_score,f1_score,recall_score,auc

def metric_scores(y_values_all,probas_all,predictions_all):

    #print(y_values_all)
    #print(predictions_all)

    aucs = [roc_auc_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    accs = [accuracy_score(y, pred) for y, pred in zip(y_values_all, predictions_all)]

    for prob_i in probas_all:
        # print(prob_i)
        for i in range(len(prob_i)):
            # print(prob_i[i])
            if prob_i[i] > 0.5:
                prob_i[i] = 1
            else:
                prob_i[i] = 0


    precision_scores = [precision_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    recall_scores = [recall_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    f1_scores = [f1_score(y, proba) for y, proba in zip(y_values_all, probas_all)]

    pr_all = []
    for i in range(len(predictions_all)):

        precision, recall, _ = metrics.precision_recall_curve(y_values_all[i],predictions_all[i])
        pr_auc = metrics.auc(recall, precision)
        pr_all.append(pr_auc)

    #pr_auc = metrics.auc(np.mean(recall_scores), np.mean(precision_scores))

    print('accuracy: ',np.mean(accs),accs)
    print('roc_auc :',np.mean(aucs),aucs)
    print('pr_auc: ',np.mean(pr_auc),pr_all)
    print('precision_scores: ',np.mean(precision_scores),precision_scores)
    print('recall scores: ',np.mean(recall_scores),recall_scores)
    print('f1_scores: ',np.mean(f1_scores),f1_scores)


