import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, dataset, out_dim=None, dropout=None, JK=None, double_linear=False):
        super().__init__()
        self.dropout = 0.6 if dropout is None else dropout
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=self.dropout)
        out_dim = dataset.num_classes if out_dim is None else out_dim
        self.conv2 = GATConv(8*8, out_dim, heads=1, dropout=self.dropout, concat=False)
        self.JK = JK
        if self.JK == 'cat':
            if double_linear:
                self.JK_proj = nn.Sequential(
                    nn.Linear(dataset.num_features, out_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(out_dim * 2, out_dim)
                )
            else:
                self.JK_proj = nn.Linear(dataset.num_features, out_dim)
            

    def forward(self, data):
        inp, edge_index = data.x, data.edge_index
        x = F.dropout(inp, self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.JK == 'cat':
            tmp = self.JK_proj(inp)
            x = torch.cat([tmp, x], dim=-1)
        return x
