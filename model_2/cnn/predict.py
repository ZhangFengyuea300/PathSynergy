import torch.utils.data as Data
import pandas as pd
import torch
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from model_function import *
from torch_geometric.loader import DataLoader
from CNN import CNN
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F

def predicting(model, device,loader_test1, loader_test2):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()

    with torch.no_grad():
        for data in zip(loader_test1, loader_test2):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data2.view(-1,1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

x = np.load('matrix_drug_cell.npy')
df_test = pd.read_csv('6271_drug_synergy_CNN.csv')
y = np.array(df_test['label'])  # 标签是0或1，形状是(300,)
print("y",y)

x_test = torch.tensor(x, dtype=torch.float)
y_test = torch.tensor(y, dtype=torch.int64)
# print("y_test",y_test)
batch_size = 64
# test_torch_dataset = Data.TensorDataset(x_test, y_test)

test_loader1 = DataLoader(dataset=x_test,batch_size=batch_size,shuffle=None)
test_loader2 = DataLoader(dataset=y_test,batch_size=batch_size,shuffle=None)

# CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load model
model = CNN().to(device)
#model.load_state_dict(torch.load('CNN.pt'))
model_name = "CNN"
path = "trained_model/" + model_name
try:
    model.load_state_dict(torch.load(path))
except:
    print("Wrong model type!")
    sys.exit()


y_true, prob, y_pred = predicting(model, device, test_loader1, test_loader2)
print(y_true)
print("\nModel predictions: ")
for i, row in df_test.iterrows():
    print(
        "{} drug1: {}, drug2: {}, cell: {}, True label: {} | Prediction: {:.0f} (score={:.3f})".format(
            i + 1,
            row.drug1,
            row.drug2,
            row.cell,
            row.label,
            y_pred[i],
            prob[i],
        )
    )


    # 保存每次训练的结果
    file_results = 'result/predict' +  '.txt'
    with open(file_results, 'a') as f:
        f.write(f'i: {i}, drug1: {row.drug1}, drug2: {row.drug2},cell: {row.cell}, True label: {row.label}, Prediction: {y_pred[i]}, score: {prob[i]}\n')
        
acc = accuracy_score(y_true, y_pred)
print("acc:",acc)


df_pred = df_test.copy()
df_pred["prediction"] = y_pred
df_pred["probability"] = prob
j_pred = df_pred.to_json(orient="records")

n_ones_true = len(df_pred[df_pred.label == 1])
n_ones_pred = len(df_pred[df_pred.prediction == 1])
ncorrect = df_pred[df_pred.prediction == df_pred.label].prediction.count()
print("\nNumber of 1s: True={}, Predicted={}".format(n_ones_true, n_ones_pred))
print(
    "Number of 0s: True={}, Predicted={}".format(
        len(df_pred) - n_ones_true, len(df_pred) - n_ones_pred
    )
)
print(
    "\Correct predictions: {}/{} = {:.2%}".format(
        ncorrect, len(df_pred), ncorrect / len(df_pred)
    )
)

# write predictions to disk
with open("result/predictions", "w") as f:
    f.write(j_pred)
print("\nPredictions written to result/predictions.json \n")


