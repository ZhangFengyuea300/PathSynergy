#函数
import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_score,f1_score,recall_score,auc


def train(model, device, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []
    for x_train, y_train in dataloader:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        output = model(x_train)
        #print("output:",output)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()

        outputs = output.argmax(dim=1)
 
        total_loss += loss.item()
        y_true.append(y_train.detach().cpu().numpy())
        y_pred.append(outputs.detach().cpu().numpy())
 
    train_acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    train_auc = roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))
    
    return total_loss / len(dataloader), train_acc, train_auc
 

def test(model,device, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
 
            outputs = output.argmax(dim=1)
            total_loss += loss.item()
            y_true.append(target.detach().cpu().numpy())
            y_pred.append(outputs.detach().cpu().numpy())
 
    test_acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    test_auc = roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))
   
    return total_loss / len(dataloader), test_acc, test_auc



def predicting(model, device, loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader_test.dataset)))
    with torch.no_grad():
        for data, y in loader_test:
            data = data.to(device)
            output = model(data)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y.view(-1,1)), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()




def metric_scores(y_values_all,probas_all,predictions_all):

    #print(y_values_all)
    #print(predictions_all)

    aucs = [roc_auc_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    accs = [accuracy_score(y, pred) for y, pred in zip(y_values_all, predictions_all)]

    for prob_i in probas_all:
        # print(prob_i)
        for i in range(len(prob_i)):
            print(prob_i[i])
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

def metrics_draw(true,pred,name):

    true = true.detach().numpy()
    pred = pred.detach().numpy()

    Truelist = []
    Problist = []
    for i in range(len(true)):
        Truelist.append(true[i][0])
        Problist.append(pred[i][0])

    Problist_int = []
    for i in range(len(Problist)):
        if Problist[i] >= 0.5:
            Problist_int.append(1)
        else:
            Problist_int.append(0)
    #print(Problist_int)


    precision_scores = metrics.precision_score(Truelist,Problist_int)
    recall_scores = metrics.recall_score(Truelist,Problist_int)
    f1_scores = metrics.f1_score(Truelist,Problist_int)
    print('f1_scores:', f1_scores)
    print('precision_scores:', precision_scores)
    print('recall_scores:', recall_scores)

    with open('metrics.txt','a') as f:
        f.write('f1_scores:')
        f.write(str(f1_scores))
        f.write('\r\n')
        f.write('precision_scores:')
        f.write(str(precision_scores))
        f.write('\r\n')
        f.write('recall_scores:')
        f.write(str(recall_scores))
        f.write('\r\n')


    precision,recall,_ = metrics.precision_recall_curve(Truelist,Problist)
    pr_auc = metrics.auc(recall,precision)
    print('pr_auc:', pr_auc)

    plt.figure(1)
    plt.plot(recall, precision, 'g', label='AUPR = %0.4f' % pr_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.savefig('./%sAUPR'%name+'.jpg')

    fpr, tpr, thresholds = metrics.roc_curve(Truelist, Problist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print('roc_auc:',roc_auc)

    plt.figure(2)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig('./%sAUC'%name+'.jpg')
    #plt.show()


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')