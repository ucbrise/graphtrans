import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gnn_module import GNNNodeEmbedding
from modules.transformer_encoder import TransformerNodeEncoder
from .base_model import BaseModel

import numpy as np

class GNNTransformer(BaseModel):
    @staticmethod
    def add_args(parser):
        TransformerNodeEncoder.add_args(parser)

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.gnn = GNNNodeEmbedding(args.gnn_virtual_node, args.gnn_num_layer,
                args.gnn_emb_dim, node_encoder, edge_encoder_cls, 
                JK=args.gnn_JK, drop_ratio=args.gnn_dropout, 
                residual=args.gnn_residual, gnn_type=args.gnn_type)

        self.gnn2transformer = nn.Linear(args.gnn_emb_dim, args.d_model)
        self.transformer_encoder = TransformerNodeEncoder(args)

        self.num_tasks = num_tasks
        self.pooling = args.graph_pooling
        self.graph_pred_linear_list = torch.nn.ModuleList()

        self.max_seq_len = args.max_seq_len
        output_dim = args.d_model
        if args.max_seq_len is None:
            self.graph_pred_linear = torch.nn.Linear(output_dim, self.num_tasks)
        else:
            for i in range(args.max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(output_dim, self.num_tasks))

    def forward(self, batched_data, perturb=None):
        h_node = self.gnn(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)
        transforemr_out, mask = self.transformer_encoder(h_node, batched_data.batch) # [s, b, h], [b, s]

        if self.pooling == 'last':
            h_graph = transforemr_out[-1]
        elif self.pooling == 'mean':
            h_graph = transforemr_out.sum(0) / mask.sum(-1, keepdim=True)
        else:
            raise NotImplementedError

        if self.max_seq_len is None:
            return self.graph_pred_linear(h_graph)
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list
        
