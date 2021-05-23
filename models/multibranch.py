from .base_model import BaseModel
from modules.multibranch_module import MultiBranchNode
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from modules.gnn_module import GNNNodeEmbedding
from modules.transformer_encoder import TransformerNodeEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MultiBranch(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim + args.d_model
    
    @staticmethod
    def add_args(parser):
        TransformerNodeEncoder.add_args(parser)
        MultiBranchNode.add_args(parser)
    @staticmethod
    def name(args):
        name = f'{args.model_type}-pooling={args.graph_pooling}'
        name += f'+{args.gnn_type}'
        name += '-virtual' if args.gnn_virtual_node else ''
        name += f'-d={args.d_model}'
        name += f'-tdp={args.transformer_dropout}'
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        gnn_node = GNNNodeEmbedding(args.gnn_virtual_node, args.gnn_num_layer,
                args.gnn_emb_dim, node_encoder, edge_encoder_cls, 
                JK=args.gnn_JK, drop_ratio=args.gnn_dropout, 
                residual=args.gnn_residual, gnn_type=args.gnn_type)
        transformer_encoder = TransformerNodeEncoder(args)

        self.multibranch = MultiBranchNode(gnn_node, transformer_encoder, args)

        self.emb_dim = self.multibranch.emb_dim
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
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, 2*self.emb_dim), torch.nn.BatchNorm1d(2*self.emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*self.emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(self.emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.max_seq_len is None:
            if self.graph_pooling == "set2set":
                self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            if self.graph_pooling == "set2set":
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(2*self.emb_dim, self.num_tasks))
            else:
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(self.emb_dim, self.num_tasks))

    def forward(self, batched_data, perturb=None):
        h_node = self.multibranch(batched_data, perturb)

        h_graph = self.pool(h_node, batched_data.batch)

        if self.max_seq_len is None:
            return self.graph_pred_linear(h_graph)
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list
