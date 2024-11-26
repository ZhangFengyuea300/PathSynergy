import numpy as np
import torch
import torch.nn as nn
from codes.train_pipeline import *
from models.gat import GATNet
import os
import random
from torch_geometric.loader import DataLoader
from codes.create_data import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix,cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score,f1_score
import warnings
warnings.filterwarnings("ignore")

cellfile = "cell_line_GAT"
datafile = "6271_drug_synergy_GAT"

# CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nThe code uses GPU...")
else:
    device = torch.device("cpu")
    print("\nThe code uses CPU!!!")

creat_data(datafile, cellfile)
#creat_data(testfile, cellfile)
drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
#print('drug1_data', drug1_data)
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

# five-fold cross-validation
lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)

# split train, validation and test data
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
    print(f'i_time: {i_time+1}')
    test_num = random_num[pot * i_time:pot * (i_time + 1)]
    train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

    TRAIN_BATCH_SIZE = 64    
    TEST_BATCH_SIZE = 64

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    #print(drug1_data_test.y)
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)


    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    model = GATNet
    model = model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)          

    #model_file_name = 'result/GAT' + str(i) + '--model_' + datafile +  '.model'
    #result_file_name = 'result/GAT' + str(i) + '--result_' + datafile + '.csv'
    file_AUCs = 'result/GAT' + str(i_time) + '--AUCs--' + datafile + '.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\tf1_score')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')


    NUM_EPOCHS = 200
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        reallabel, predictscore, predictlabel = predicting(model, device, drug1_loader_test, drug2_loader_test)

        # compute preformence
        AUC = roc_auc_score(reallabel, predictscore)
        precision, recall, threshold = metrics.precision_recall_curve(reallabel, predictscore)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(reallabel, predictlabel)
        tn, fp, fn, tp = confusion_matrix(reallabel, predictlabel).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(reallabel, predictlabel)
        ACC = accuracy_score(reallabel, predictlabel)
        KAPPA = cohen_kappa_score(reallabel, predictlabel)
        recall = recall_score(reallabel, predictlabel)
        f1_scores = f1_score(reallabel, predictlabel)

        def save_AUCs(AUCs, filename):
            with open(filename, 'a') as f:
                f.write('\t'.join(map(str, AUCs)) + '\n')

        # if best_auc < AUC:
        #     best_auc = AUC
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall, f1_scores]
        save_AUCs(AUCs, file_AUCs)
    
        # print('best_auc', best_auc)
    
 

# # create model
# print(model)
# model_params = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_params])
# print(f"\nNumber of trainable parameterss: {params}\n")


# # save trained model
# torch.save(model,'GAT.pt')
# print('Save model!')


# # save trained model

# if not os.path.exists("trained_model"):
#     os.makedirs("trained_model")

# model_name = "GAT" 
# path = "trained_model/" + model_name 
# print("Saving trained model to {}".format(path))
# torch.save(model.state_dict(), path)

