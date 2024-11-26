#函数
import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_score,f1_score,recall_score,auc


# 训练函数
def train_gc(model, device, combined_gat_loader_train,train_loader,model_a,model_b,optimizer, loss_fn):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []
    for (data1, data2),(x_train,y_train) in zip(combined_gat_loader_train,train_loader):
        # 将GAT数据移动到设备
        data1 = data1.to(device)
        data2 = data2.to(device)
        GAT_output = model_a(data1, data2)
        #print("GAT_output:", GAT_output)
        # 将CNN数据移动到设备（例如GPU）
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        CNN_output = model_b(x_train)
        #print("CNN_output:", CNN_output)
        optimizer.zero_grad()
        # 将GAT_output和CNN_output作为输入传递给融合模型
        output = model(GAT_output, CNN_output)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
        outputs = output.argmax(dim=1)
        total_loss += loss.item()
        y_true.append(y_train.detach().cpu().numpy())
        y_pred.append(outputs.detach().cpu().numpy())
    
 
    train_acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    train_auc = roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))
        
    return total_loss / len(train_loader), train_acc, train_auc


# 测试函数
def test_gc(model,device, combined_gat_loader_test,test_loader,model_a,model_b, loss_fn):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for (data1, data2),(x_test,y_test) in zip(combined_gat_loader_test,test_loader):
            # 将GAT数据移动到设备
            data1 = data1.to(device)
            data2 = data2.to(device)
            GAT_output = model_a(data1, data2)
            #print("GAT_output:", GAT_output)
            # 将CNN数据移动到设备（例如GPU）
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            CNN_output = model_b(x_test)
            #print("CNN_output:", CNN_output)
            output = model(GAT_output, CNN_output)
            loss = loss_fn(output, y_test)
            outputs = output.argmax(dim=1)
            total_loss += loss.item()

            y_true.append(y_test.detach().cpu().numpy())
            y_pred.append(outputs.detach().cpu().numpy())
 
    test_acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    test_auc = roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))
   
    return total_loss / len(test_loader), test_acc, test_auc   
    



# 预测函数
def predict_gc(model, device, combined_gat_loader_test,test_loader,model_a,model_b):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    #print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for (data1, data2),(x_test,y_test) in zip(combined_gat_loader_test,test_loader):
            # 将GAT数据移动到设备
            data1 = data1.to(device)
            data2 = data2.to(device)
            GAT_output = model_a(data1, data2)
            #print("GAT_output:", GAT_output)
            # 将CNN数据移动到设备（例如GPU）
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            CNN_output = model_b(x_test)
            # print("CNN_output:", CNN_output)
            output = model(GAT_output, CNN_output)
            #print("output:", output)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y_test.view(-1,1).cpu()), 0)
            
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()





def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
