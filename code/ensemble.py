
import torch
import torch.nn as nn



class EnsembleModel(nn.Module):
    def __init__(self,model_A,model_B):
        super(EnsembleModel, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        # 融合参数
        self.fusion_parameter = nn.Parameter(torch.tensor(0.5,requires_grad=True))
 
    def forward(self, *inputs):
        output_A, output_B = inputs
        # 使用融合参数来融合两个模型的输出
        return self.fusion_parameter * output_A + (1 - self.fusion_parameter) * output_B

