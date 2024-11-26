import torch.optim as optim
import torch
import torch.nn as nn
from model_2.cnn import CNN as MAP_CNN
from model_1.codes import GAT as CELL_GAT
import numpy as np
import pandas as pd
import os
from ensemble import EnsembleModel
from ensemble_function import *
import random
import torch.utils.data as Data
from model_1.codes.train_pipeline import *
from model_1.codes.create_data import *
from torch_geometric.loader import DataLoader


# CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nThe code uses GPU...")
else:
    device = torch.device("cpu")
    print("\nThe code uses CPU!!!")

### 加载数据集 ###############################################
# 加载modelA的数据
cellfile = "cell_line_GAT"  # GAT细胞系数据
datafile = "6271_drug_synergy_GAT"  # GAT药物样本数据
creat_data(datafile, cellfile)   # GAT创建数据

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

# 加载modelB的数据
x = np.load('./data/data_CNN/matrix_drug_cell.npy')   # 通路背景数据
label = pd.read_csv('./data/data_CNN/6271_drug_synergy_CNN.csv')
y = np.array(label['label'])  # 标签是0或1，形状是(300,) 

# ### 初始化模型 #############################################
# 假设 model_a 和 model_b 已经被定义和初始化
model_a = CELL_GAT()
model_b = MAP_CNN()
model_a.load_state_dict(torch.load('GAT'))
model_b.load_state_dict(torch.load('CNN'))

# 实例化融合模型
fused_model = EnsembleModel
fused_model = fused_model(model_a,model_b).to(device)
optimizer = optim.Adam(fused_model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

# ### 使用训练和测试函数 #########################################
lenth = len(x)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

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
    print(f'i_time: {i_time+1}')
    # CNN数据处理  ###################################
    test_num = random_num[pot * i_time:pot * (i_time + 1)]
    train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

    epoch = 200
    for epoch in range(epoch):
        x_train = x[train_num]
        x_test = x[test_num]
        x_train = torch.tensor(x_train, dtype=torch.float)
        x_test = torch.tensor(x_test, dtype=torch.float)

        y_train = y[train_num]
        y_test = y[test_num]
        y_train =  torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
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


        # GAT数据处理  ##########################################
        TRAIN_BATCH_SIZE = 64
        TEST_BATCH_SIZE = 64

        drug1_data_train = drug1_data[train_num]
        drug1_data_test = drug1_data[test_num]
        drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        drug2_data_test = drug2_data[test_num]
        drug2_data_train = drug2_data[train_num]
        drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


        # 确保两个GAT的DataLoader长度相同
        assert len(drug1_loader_train ) == len(drug2_loader_train), "GAT train DataLoaders must have the same length"
        assert len(drug1_loader_test ) == len(drug2_loader_test), "GAT test DataLoaders must have the same length"

        # 合并两个GAT的DataLoader
        combined_gat_loader_train = zip(drug1_loader_train, drug2_loader_train)
        combined_gat_loader_test = zip(drug1_loader_test, drug2_loader_test)


        train_loss,train_acc,train_auc = train_gc(fused_model,device,combined_gat_loader_train,train_loader,model_a,model_b, optimizer, loss_fn)
        test_loss,test_acc,test_auc= test_gc(fused_model,device, combined_gat_loader_test,test_loader,model_a,model_b, loss_fn)
        

        print('i_time: ', i_time, 'Epoch: ', epoch, '|train_loss: ',train_loss , '| accuracy_train: ',
                    train_acc,'|test_loss: ',test_loss , '| accuracy_test: ', test_acc)

        
        # 保存每次训练的结果
        file_results = 'result/results_GAT_CNN' + '_' + str(i_time) + '.txt'
        with open(file_results, 'a') as f:
            f.write(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},'
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n')
        

# 保存模型为.pt文件
torch.save(fused_model.state_dict(), 'GAT_CNN.pt')
print('Save model!')


if not os.path.exists("trained_model"):
    os.makedirs("trained_model")

model_name = "GAT_CNN" 
path = "trained_model/" + model_name 
print("Saving trained model to {}".format(path))
torch.save(fused_model.state_dict(), path)

