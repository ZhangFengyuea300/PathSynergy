#  增加F1_score指标
import torch
import torch.nn as nn
import torch.optim as optim
from CNN import CNN
from model_function import *
import os
import random
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score,f1_score


# CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nThe code uses GPU...")
else:
    device = torch.device("cpu")
    print("\nThe code uses CPU!!!")

x = np.load('matrix_drug_cell.npy')
#x = torch.tensor(data, dtype=torch.float)
#print(x.shape)
#x = load_data(data, batch_size=128, train_split=0.9) 
label = pd.read_csv('6271_drug_synergy_CNN.csv')
y = np.array(label['label']) 
#y = torch.tensor(y, dtype=torch.long)
#print(y.shape)

# data and model
model = CNN
model = model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
 
# train and test
lenth = len(x)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

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

    #train data 、test data
    x_train = x[train_num]
    x_test = x[test_num]
    x_train = torch.tensor(x_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)

    y_train = y[train_num]
    y_test = y[test_num]
    #print(y_test)

    y_train =  torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # y_train = torch.reshape(y_train, (-1, 1))
    # y_test = torch.reshape(y_test, (-1, 1))

    batch_size = 64
    train_torch_dataset = Data.TensorDataset(x_train, y_train)
    test_torch_dataset = Data.TensorDataset(x_test, y_test)
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print(f'i_time: {i_time+1}')
    
    epoch =200
    for epoch in range(epoch):
        train_loss,train_acc,train_auc = train(model,device, train_loader, optimizer, loss_fn)
        test_loss,test_acc,test_auc= test(model,device, test_loader, loss_fn)
        T, S, Y = predicting(model, device, test_loader)  # T is correct label；S is predict score；Y is predict label
        
        # compute preformence
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        fpr, tpr, thresholds = metrics.roc_curve(T, S, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        RECALL = recall_score(T, Y)
        f1_scores = f1_score(T, Y)
        # AUC
        file_AUCs = 'result/model/CNN1/CNN_train' + '_' + str(i_time) + '--AUCs--' + '.txt'
        AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\tf1_scores')
        with open(file_AUCs, 'a') as f:
            f.write(AUCs + '\n')
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL, f1_scores]
        save_AUCs(AUCs, file_AUCs)

        print('i_time: ', i_time, 'Epoch: ', epoch, '|train_loss: ',train_loss , '| accuracy_train: ',
                    train_acc,'|test_loss: ',test_loss , '| accuracy_test: ', test_acc)
  
        # save results
        file_results = 'result/model/CNN1/results_train' + '_' + str(i_time) + '.txt'
        with open(file_results, 'a') as f:
            f.write(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},'
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n')
        


# # save trained model
# torch.save(model.state_dict(), 'CNN.pt')
# print('Save model!')






