import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(887 * 11 * 4 * 4, 300)
        self.fc2 = nn.Linear(300, 84)
        self.fc3 = nn.Linear(84, 2)  # 二分类输出
        self.dropout = nn.Dropout(0.2) # 定义Dropout层，用于正则化
        self.softmax = nn.Softmax(dim=1) # 定义softmax激活函数，用于多分类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 展平多维度张量至一维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# 加载预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)  # 使用你定义的CNN类
model.load_state_dict(torch.load('CNN'))  # 确保文件名和路径正确
model.eval()

# 检查模型的层名称
for name, layer in model.named_modules():
    print(name)

# 选择一个目标卷积层（根据打印出的层名称选择）
target_layer_name = 'conv1'  # 更改为你的模型中的适当层名称
target_layer = dict([*model.named_modules()])[target_layer_name]

# 定义一个函数，用于计算梯度
def compute_gradients(input_data, target_class):
    input_data.requires_grad_(True)
    output = model(input_data)
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()
    return input_data.grad.data.abs()

# 初始化输入数据
data = np.load('./data/top_1.npy')
input_data = torch.from_numpy(data)

# 调整输入数据的维度
input_data = input_data.view(1, 3, 58, 3562)  # 确保维度为 [batch_size, channels, height, width]
input_data = input_data.to(device).float()  # 确保输入数据类型为float并移动到设备
target_class = 1

# 计算梯度
gradients = compute_gradients(input_data, target_class)

# 调整图像数据的形状
gradients_np = gradients.squeeze().detach().cpu().numpy()
gradients_np = np.transpose(gradients_np, (1, 2, 0))  # 转换为 (height, width, channels)

# 将梯度值展平并排序
flat_gradients = gradients_np.flatten()
indices = np.argsort(flat_gradients)[::-1]  # 从高到低排序

# 输出按梯度值从高到低的所有点的坐标
for idx in indices:
    y, x, _ = np.unravel_index(idx, gradients_np.shape)
    print(f"坐标: ({y}, {x}), 梯度值: {flat_gradients[idx]}")