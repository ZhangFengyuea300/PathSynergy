#机器学习方法预测
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from machine_function import metric_scores

# load data
data = pd.read_csv('./mult_model/machine/independent_machine/independent_feature.csv')
train_data = pd.read_csv('./mult_model/machine/feature.csv')
y = data['label']
X = data.drop(['label'],axis=1)
print(X.shape,y.shape)
X_train = train_data.drop(['label'],axis=1)
y_train = train_data['label']

X = np.array(X)
y = np.array(y)
X_train = np.array(X_train)
y_train = np.array(y_train)

clf_svm = SVC(probability=True)
clf_rf = RandomForestClassifier()
clf_gbm = GradientBoostingClassifier()
clf_xgb = XGBClassifier()
xgb.set_config(verbosity=0)

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

X_train = X_train
y_train = y_train
X_test = X
y_test = y

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

print(probas_svm)
print(y_values_svm)

import csv
for j in range(len(probas_svm)):
    # 写入 SVM 文件
    with open('indepredent_svm.csv', 'a', newline='') as f_svm:
        writer = csv.writer(f_svm)
        for proba, true_label in zip(probas_svm[j], y_values_svm[j]):
            writer.writerow([proba, true_label])

    # 写入 RF 文件
    with open('indepredent_rf.csv', 'a', newline='') as f_rf:
        writer = csv.writer(f_rf)
        for proba, true_label in zip(probas_rf[j], y_values_rf[j]):
            writer.writerow([proba, true_label])

    # 写入 GBM 文件
    with open('indepredent_gbm.csv', 'a', newline='') as f_gbm:
        writer = csv.writer(f_gbm)
        for proba, true_label in zip(probas_gbm[j], y_values_gbm[j]):
            writer.writerow([proba, true_label])

    # 写入 XGB 文件
    with open('indepredent_xgb.csv', 'a', newline='') as f_xgb:
        writer = csv.writer(f_xgb)
        for proba, true_label in zip(probas_xgb[j], y_values_xgb[j]):
            writer.writerow([proba, true_label])

metric_scores(y_values_svm, probas_svm, predictions_svm)
metric_scores(y_values_rf, probas_rf, predictions_rf)
metric_scores(y_values_gbm, probas_gbm, predictions_gbm)
metric_scores(y_values_xgb, probas_xgb, predictions_xgb)