#predict_4905_linux
#预测函数————超算
import torch
import torch.nn.functional as F
from cnn import CNN as MAP_CNN
from gat import GAT as CELL_GAT
import numpy as np
import pandas as pd
from ensemble import EnsembleModel
import torch.utils.data as Data
from train_pipeline import *
from create_data import *
from torch_geometric.loader import DataLoader


# 预测函数
def predict_gc(model, device, combined_gat_loader_test,test_loader,model_a,model_b):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    with torch.no_grad():
        for (data1, data2),(data) in zip(combined_gat_loader_test,test_loader):
            # 将GAT数据移动到设备
            data1 = data1.to(device)
            data2 = data2.to(device)
            GAT_output = model_a(data1, data2)
            # 将CNN数据移动到设备
            data_1 = data[0]
            data_2 = data[1]
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            CNN_output = model_b(data_1)
            output = model(GAT_output, CNN_output)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data_2.view(-1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


# CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nThe code uses GPU...")
else:
    device = torch.device("cpu")
    print("\nThe code uses CPU!!!")

# ### 加载数据集 ######################################################
# 加载modelA的数据
cellfile = "cell_line_GAT"    # GAT细胞系数据
datafile = "GAT_drug_synergy_4905"   # GAT药物样本数据
creat_data(datafile, cellfile)     # GAT创建数据

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

# 加载modelB的数据
x = np.load('./data/CNN_4905/matrix_drug_cell_4905.npy')     # 通路背景数据
label = pd.read_csv('./data/CNN_4905/CNN_drug_Sorafenib_4905.csv')  # 样本数据
y = np.array(label['label'])  

# ############## 初始化模型 #############################################
# 假设 model_a 和 model_b 已经被定义和初始化
model_a = CELL_GAT()
model_b = MAP_CNN()
model_a.load_state_dict(torch.load('GAT.pt'))
model_b.load_state_dict(torch.load('CNN.pt'))

# 实例化融合模型
fused_model = EnsembleModel(model_a,model_b).to(device)
fused_model.load_state_dict(torch.load('GAT_CNN'))



x_test = torch.tensor(x, dtype=torch.float)
y_test = torch.tensor(y, dtype=torch.int64)
batch_size = 64
test_loader1 = Data.DataLoader(dataset=x_test,batch_size=batch_size,shuffle=None)
test_loader2 = Data.DataLoader(dataset=y_test,batch_size=batch_size,shuffle=None)
test_loader = zip(test_loader1, test_loader2)    # 合并两个CNN的DataLoader

# GAT数据处理  ##########################################
drug1_data_test = drug1_data
drug2_data_test = drug2_data
drug1_loader_test = DataLoader(drug1_data_test, batch_size = 64, shuffle = True)
drug2_loader_test = DataLoader(drug2_data_test, batch_size = 64, shuffle = True)

assert len(drug1_loader_test ) == len(drug2_loader_test), "GAT test DataLoaders must have the same length"    # 确保两个GAT的DataLoader长度相同
combined_gat_loader_test = zip(drug1_loader_test, drug2_loader_test)     # 合并两个GAT的DataLoader


reallabel, predictscore, predictlabel = predict_gc(fused_model,device, combined_gat_loader_test,test_loader,model_a,model_b)

result = [list(map(str, reallabel)), list(map(str, predictlabel)), list(map(float, predictscore))]
#result = [reallabel, predictlabel, predictscore]

filename = "SORAFENIB_4905.csv" 
with open(filename, 'a') as f:
    f.write('\t'.join(map(str,result)) + '\n')