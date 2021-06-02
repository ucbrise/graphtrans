import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GlobalAttention,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from modules.gnn_module import GNNNodeEmbedding
from modules.transformer_encoder import TransformerNodeEncoder
from modules.utils import pad_batch, unpad_batch

from .base_model import BaseModel


class Transformer(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.d_model

    @staticmethod
    def add_args(parser):
        TransformerNodeEncoder.add_args(parser)

    @staticmethod
    def name(args):
        name = f"{args.model_type}-pooling={args.graph_pooling}"
        name += f"+{args.gnn_type}"
        name += "-virtual" if args.gnn_virtual_node else ""
        name += f"-d={args.d_model}"
        name += f"-tdp={args.transformer_dropout}"
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.transformer = TransformerNodeEncoder(args)

        self.node_encoder = node_encoder

        self.emb_dim = args.d_model
        self.num_tasks = num_tasks
        self.max_seq_len = args.max_seq_len
        self.graph_pooling = args.graph_pooling

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
                    torch.nn.Linear(self.emb_dim, 2 * self.emb_dim),
                    torch.nn.BatchNorm1d(2 * self.emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.emb_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(self.emb_dim, processing_steps=2)
        elif self.graph_pooling == "cls":
            self.pool = None
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.max_seq_len is None:
            if self.graph_pooling == "set2set":
                self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            if self.graph_pooling == "set2set":
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(2 * self.emb_dim, self.num_tasks))
            else:
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(self.emb_dim, self.num_tasks))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None
        encoded_node = (
            self.node_encoder(x)
            if node_depth is None
            else self.node_encoder(
                x,
                node_depth.view(
                    -1,
                ),
            )
        )
        tmp = encoded_node + perturb if perturb is not None else encoded_node

        h_node, src_key_padding_mask, num_nodes, mask, max_num_nodes = pad_batch(tmp, batch, self.transformer.max_input_len, get_mask=True)
        h_node, src_key_padding_mask = self.transformer(h_node, src_key_padding_mask)
        if self.graph_pooling == "cls":
            h_graph = h_node[-1]
        else:
            h_node = unpad_batch(h_node, tmp, num_nodes, mask, max_num_nodes)
            h_graph = self.pool(h_node, batched_data.batch)

        if self.max_seq_len is None:
            return self.graph_pred_linear(h_graph)
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list
