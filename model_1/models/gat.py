import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Softmax, Dropout
from torch.nn import Tanh, Sigmoid, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT model
class GATNet(torch.nn.Module):
    def __init__(
        self,
        num_features_xd=78,
        n_output=2,
        num_features_xt=954,
        output_dim=128,
        dropout=0.2,
        file=None,
    ):
        super(GATNet, self).__init__()

        # graph drug layers
        self.drug_gat1 = GATConv(num_features_xd, output_dim, heads=10,)
        self.drug_gat2 = GATConv(output_dim * 10, output_dim,)      
        self.drug_fc_g1 = nn.Linear(output_dim, output_dim)
        self.filename = file

        # DL cell featrues  MLP
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            #nn.Tanh(),
            #nn.Sigmoid(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            #nn.Tanh(),
            #nn.Sigmoid(),
            nn.Linear(512, output_dim * 2),
            nn.ReLU(),
            #nn.Tanh(),
            #nn.Sigmoid(),
        )

        # combined layers 
        self.fc1 = nn.Linear(output_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization  
        self.rule = nn.ReLU()
        #self.rule = nn.Tanh()
        #self.rule = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        self.softmax = nn.Softmax(dim=1) # softmax

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = (
            data1.x,
            data1.edge_index,
            data1.batch,
            data1.cell,
        )

        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1 = self.drug_gat1(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.0, training=self.training)
        x1 = self.drug_gat2(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.0, training=self.training)
        x1 = gmp(x1, batch1)  # global max pooling
        x1 = self.drug_fc_g1(x1)
        x1 = self.rule(x1)

        # deal drug2
        x2 = self.drug_gat1(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.0, training=self.training)
        x2 = self.drug_gat2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.0, training=self.training)
        x2 = gmp(x2, batch2)  # global max pooling
        x2 = self.drug_fc_g1(x2)
        x2 = self.rule(x2)

        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)

        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        xc = F.normalize(xc, 2, 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.rule(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.rule(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.rule(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.softmax(out)

        return out
