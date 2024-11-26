import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,train_test_split,KFold,StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
from machine_function import metric_scores
import torch
import random
import os

# load data
data = pd.read_csv('./mult_model/machine/feature.csv')

y = data['label']
X = data.drop(['label'],axis=1)
print(X.shape,y.shape)

X = np.array(X)
y = np.array(y)


clf_svm = SVC(probability=True)
clf_rf = RandomForestClassifier()
clf_gbm = GradientBoostingClassifier()
clf_xgb = XGBClassifier()
xgb.set_config(verbosity=0)

#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
lenth = len(X)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)


y_values_svm = []
predictions_svm = []
probas_svm = []

y_values_rf = []
predictions_rf = []
probas_rf = []

y_values_gbm = []
predictions_gbm = []
probas_gbm = []

y_values_xgb = []
predictions_xgb = []
probas_xgb = []

tprs_svm = []
aucs_svm = []
mean_fpr = np.linspace(0, 1, 100)


#随机样本
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)
random_num = random.sample(range(0, lenth), lenth)
for i_time in range(5):
    test_num = random_num[pot * i_time:pot * (i_time + 1)]
    train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

    X_train = X[train_num]
    y_train = y[train_num]
    X_test = X[test_num]
    y_test = y[test_num]
    
    # svm
    clf_svm.fit(X_train, y_train)
    predictions_svm.append(clf_svm.predict(X_test))
    probas_svm.append(clf_svm.predict_proba(X_test).T[1])  # Probabilities for class 1
    y_values_svm.append(y_test)
    print("svm:",y_test)

    # randomforest
    clf_rf.fit(X_train, y_train)
    predictions_rf.append(clf_rf.predict(X_test))
    probas_rf.append(clf_rf.predict_proba(X_test).T[1])  # Probabilities for class 1    
    y_values_rf.append(y_test)      
    print("rf:",y_test)

    # GBM
    clf_gbm.fit(X_train, y_train)
    predictions_gbm.append(clf_gbm.predict(X_test)) 
    probas_gbm.append(clf_gbm.predict_proba(X_test).T[1])  # Probabilities for class 1
    y_values_gbm.append(y_test)
    print("GBM:",y_test)

    # xgboost   
    clf_xgb.fit(X_train, y_train)
    predictions_xgb.append(clf_xgb.predict(X_test))
    probas_xgb.append(clf_xgb.predict_proba(X_test).T[1])  # Probabilities for class 1          
    y_values_xgb.append(y_test)
    print("xgb:",y_test)

    # # svm
    # clf_svm.fit(X[train], y[train])
    # predictions_svm.append(clf_svm.predict(X[test]))
    # probas_svm.append(clf_svm.predict_proba(X[test]).T[1])  # Probabilities for class 1
    # y_values_svm.append(y[test])
    # print("svm:",y[test])

    # # randomforest
    # clf_rf.fit(X[train], y[train])
    # predictions_rf.append(clf_rf.predict(X[test]))
    # probas_rf.append(clf_rf.predict_proba(X[test]).T[1])  # Probabilities for class 1
    # y_values_rf.append(y[test])
    # print("rf:",y[test])

    # # GBM
    # clf_gbm.fit(X[train], y[train])
    # predictions_gbm.append(clf_gbm.predict(X[test]))
    # probas_gbm.append(clf_gbm.predict_proba(X[test]).T[1])  # Probabilities for class 1
    # y_values_gbm.append(y[test])
    # print("GBM:",y[test])

    # # xgboost
    # clf_xgb.fit(X[train], y[train])
    # predictions_xgb.append(clf_xgb.predict(X[test]))
    # probas_xgb.append(clf_xgb.predict_proba(X[test]).T[1])  # Probabilities for class 1
    # y_values_xgb.append(y[test])
    # print("xgb:",y[test])

print(probas_svm)
print(y_values_svm)

for i in range(5):
    for j in range(len(probas_svm[i])):
        with open('pred_svm%s.txt'%i,'a') as f_svm:

            f_svm.write(str(probas_svm[i][j]))
            f_svm.write('\t')
            f_svm.write(str(y_values_svm[i][j]))
            f_svm.write('\n')


        with open('pred_rf%s.txt'%i,'a') as f_rf:

            f_rf.write(str(probas_rf[i][j]))
            f_rf.write('\t')
            f_rf.write(str(y_values_rf[i][j]))
            f_rf.write('\n')


        with open('pred_gbm%s.txt'%i, 'a') as f_gbm:

            f_gbm.write(str(probas_gbm[i][j]))
            f_gbm.write('\t')
            f_gbm.write(str(y_values_gbm[i][j]))
            f_gbm.write('\n')


        with open('pred_xgb%s.txt'%i, 'a') as f_xgb:

            f_xgb.write(str(probas_xgb[i][j]))
            f_xgb.write('\t')
            f_xgb.write(str(y_values_xgb[i][j]))
            f_xgb.write('\n')



metric_scores(y_values_svm, probas_svm, predictions_svm)
metric_scores(y_values_rf, probas_rf, predictions_rf)
metric_scores(y_values_gbm, probas_gbm, predictions_gbm)
metric_scores(y_values_xgb, probas_xgb, predictions_xgb)