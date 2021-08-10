import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from modules.gnn_module import GNNNodeEmbedding
from modules.masked_transformer_encoder import MaskedOnlyTransformerEncoder
from modules.transformer_encoder import TransformerNodeEncoder
from modules.utils import pad_batch, unpad_batch

from .base_model import BaseModel
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

class TransformerGNN(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim

    @staticmethod
    def add_args(parser):
        TransformerNodeEncoder.add_args(parser)
        MaskedOnlyTransformerEncoder.add_args(parser)
        group = parser.add_argument_group("GNNTransformer - Training Config")
        group.add_argument("--pretrained_gnn", type=str, default=None, help="pretrained gnn_node node embedding path")
        group.add_argument("--freeze_gnn", type=int, default=None, help="Freeze gnn_node weight from epoch `freeze_gnn`")
        group.add_argument("--graph_input_dim", type=int, default=None)

    @staticmethod
    def name(args):
        name = f"{args.model_type}-pooling={args.graph_pooling}"
        name += "-norm_input" if args.transformer_norm_input else ""
        name += f"+{args.gnn_type}"
        name += "-virtual" if args.gnn_virtual_node else ""
        name += f"-JK={args.gnn_JK}"
        name += f"-enc_layer={args.num_encoder_layers}"
        name += f"-enc_layer_masked={args.num_encoder_layers_masked}"
        name += f"-d={args.d_model}"
        name += f"-act={args.transformer_activation}"
        name += f"-tdrop={args.transformer_dropout}"
        name += f"-gdrop={args.gnn_dropout}"
        name += "-pretrained_gnn" if args.pretrained_gnn else ""
        name += f"-freeze_gnn={args.freeze_gnn}" if args.freeze_gnn is not None else ""
        name += "-prenorm" if args.transformer_prenorm else "-postnorm"
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.node_encoder = node_encoder
        self.input2transformer = nn.Linear(args.graph_input_dim, args.d_model) if args.graph_input_dim is not None else None
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        gnn_emb_dim = args.gnn_emb_dim
        self.transformer2gnn = nn.Linear(args.d_model, gnn_emb_dim)
        self.gnn_node = GNNNodeEmbedding(
            args.gnn_virtual_node,
            args.gnn_num_layer,
            args.gnn_emb_dim,
            node_encoder=None,
            edge_encoder_cls=edge_encoder_cls,
            JK=args.gnn_JK,
            drop_ratio=args.gnn_dropout,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
        )
        if args.pretrained_gnn:
            # logger.info(self.gnn_node)
            state_dict = torch.load(args.pretrained_gnn)
            state_dict = self._gnn_node_state(state_dict["model"])
            logger.info("Load GNN state from: {}", state_dict.keys())
            self.gnn_node.load_state_dict(state_dict)
        self.freeze_gnn = args.freeze_gnn

        self.num_encoder_layers = args.num_encoder_layers
        self.num_encoder_layers_masked = args.num_encoder_layers_masked

        self.num_tasks = num_tasks
        self.pooling = args.graph_pooling
        self.graph_pred_linear_list = torch.nn.ModuleList()

        self.max_seq_len = args.max_seq_len

        ### Pooling function to generate whole-graph embeddings
        if self.pooling == "sum":
            self.pool = global_add_pool
        elif self.pooling == "mean":
            self.pool = global_mean_pool
        elif self.pooling == "max":
            self.pool = global_max_pool
        elif self.pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(gnn_emb_dim, 2 * gnn_emb_dim),
                    torch.nn.BatchNorm1d(2 * gnn_emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * gnn_emb_dim, 1),
                )
            )
        elif self.pooling == "set2set":
            self.pool = Set2Set(gnn_emb_dim, processing_steps=2)
        else:
            raise ValueError(f"Invalid graph pooling type. {self.pooling}")

        if self.max_seq_len is None:
            if self.pooling == "set2set":
                self.graph_pred_linear = torch.nn.Linear(2 * gnn_emb_dim, self.num_tasks)
            else:
                self.graph_pred_linear = torch.nn.Linear(gnn_emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            if self.pooling == "set2set":
                for i in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(2 * gnn_emb_dim, self.num_tasks))
            else:
                for i in range(self.max_seq_len):
                    if args.gnn_JK == 'cat':
                        self.graph_pred_linear_list.append(torch.nn.Linear(2 * gnn_emb_dim, self.num_tasks))
                    else:
                        self.graph_pred_linear_list.append(torch.nn.Linear(gnn_emb_dim, self.num_tasks))

    def forward(self, batched_data, perturb=None):
        x = batched_data.x
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
        if self.input2transformer is not None:
            tmp = self.input2transformer(tmp)
        padded_h_node, src_padding_mask, num_nodes, mask, max_num_nodes = pad_batch(
            tmp, batched_data.batch, self.transformer_encoder.max_input_len, get_mask=True
        )  # Pad in the front
        # TODO(paras): implement mask
        transformer_out = padded_h_node
        if self.num_encoder_layers_masked > 0:
            adj_list = batched_data.adj_list
            padded_adj_list = torch.zeros((len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device)
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = self.masked_transformer_encoder(
                transformer_out.transpose(0, 1), attn_mask=padded_adj_list, valid_input_mask=src_padding_mask
            ).transpose(0, 1)
        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, b, h], [b, s]
        
        h_node = unpad_batch(transformer_out, tmp, num_nodes, mask, max_num_nodes)
        batched_data.x = self.transformer2gnn(h_node)
        h_node = self.gnn_node(batched_data, None)

        h_graph = self.pool(h_node, batched_data.batch)

        if self.max_seq_len is None:
            out = self.graph_pred_linear(h_graph)
            return out
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list

    def epoch_callback(self, epoch):
        # TODO: maybe unfreeze the gnn at the end.
        if self.freeze_gnn is not None and epoch >= self.freeze_gnn:
            logger.info(f"Freeze GNN weight after epoch: {epoch}")
            for param in self.gnn_node.parameters():
                param.requires_grad = False

    def _gnn_node_state(self, state_dict):
        module_name = "gnn_node"
        new_state_dict = dict()
        for k, v in state_dict.items():
            if module_name in k:
                new_key = k.split(".")
                module_index = new_key.index(module_name)
                new_key = ".".join(new_key[module_index + 1 :])
                new_state_dict[new_key] = v
        return new_state_dict
