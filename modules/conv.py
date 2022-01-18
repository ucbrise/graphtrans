import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.utils import degree


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim: int, edge_encoder_cls):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = edge_encoder_cls(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_encoder_cls):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = edge_encoder_cls(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(
            -1, 1
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
