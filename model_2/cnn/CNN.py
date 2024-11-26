import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Defining the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(887*11*4*4, 300)
        self.fc2 = nn.Linear(300, 84)
        self.fc3 = nn.Linear(84, 2)  # output
        self.dropout = nn.Dropout(0.2) # Dropout
        self.softmax = nn.Softmax(dim=1) # softmax

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
 