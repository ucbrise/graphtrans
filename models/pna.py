import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import (
    BatchNorm,
    GlobalAttention,
    PNAConv,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from modules.pna.pna_module import PNANodeEmbedding

from .base_model import BaseModel


class PNANet(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim

    @staticmethod
    def need_deg():
        return True

    @staticmethod
    def add_args(parser):
        PNANodeEmbedding.add_args(parser)

    @staticmethod
    def name(args):
        name = f"{args.model_type}"
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.num_layer = args.gnn_num_layer
        self.num_tasks = num_tasks
        self.max_seq_len = args.max_seq_len
        self.aggregators = args.aggregators
        self.scalers = args.scalers
        self.residual = args.gnn_residual
        self.drop_ratio = args.gnn_dropout
        self.graph_pooling = args.graph_pooling

        self.node_encoder = node_encoder

        self.pna_module = PNANodeEmbedding(node_encoder, args)

        if self.max_seq_len is None:
            self.mlp = nn.Sequential(
                nn.Linear(args.gnn_emb_dim, 35, bias=True),
                nn.ReLU(),
                nn.Linear(35, 17, bias=True),
                nn.ReLU(),
                nn.Linear(17, self.num_tasks, bias=True),
            )

        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            for i in range(self.max_seq_len):
                self.graph_pred_linear_list.append(
                    nn.Sequential(
                        nn.Linear(args.gnn_emb_dim, args.gnn_emb_dim),
                        nn.ReLU(),
                        nn.Linear(args.gnn_emb_dim, self.num_tasks),
                    )
                )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(args.gnn_emb_dim, 2 * args.gnn_emb_dim),
                    torch.nn.BatchNorm1d(2 * args.gnn_emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * args.gnn_emb_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(args.gnn_emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data, perturb=None):
        x = self.pna_module(batched_data, perturb)

        h_graph = self.pool(x, batched_data.batch)

        if self.max_seq_len is None:
            return self.mlp(h_graph)
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))
        return pred_list
