import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from modules.gnn_module import GNNNodeEmbedding
from modules.pna.pna_module import PNANodeEmbedding
from modules.transformer_encoder import TransformerNodeEncoder
from modules.utils import pad_batch

from .base_model import BaseModel


class PNATransformer(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim

    @staticmethod
    def need_deg():
        return True

    @staticmethod
    def add_args(parser):
        TransformerNodeEncoder.add_args(parser)
        PNANodeEmbedding.add_args(parser)

        group = parser.add_argument_group("GNNTransformer - Training Config")
        group.add_argument("--pretrained_gnn", type=str, default=None, help="pretrained gnn_node node embedding path")
        # group.add_argument('--drop_last_pretrained', action='store_true', default=False, help='drop the last layer for the pretrained model')
        group.add_argument("--freeze_gnn", type=int, default=None, help="Freeze gnn_node weight from epoch `freeze_gnn`")

    @staticmethod
    def name(args):
        name = f"{args.model_type}-pooling={args.graph_pooling}"
        name += "-norm_input" if args.transformer_norm_input else ""
        name += f"+{args.gnn_type}"
        name += "-virtual" if args.gnn_virtual_node else ""
        name += f"-JK={args.gnn_JK}"
        name += f"-enc_layer={args.num_encoder_layers}"
        name += f"-d={args.d_model}"
        name += f"-act={args.transformer_activation}"
        name += f"-tdrop={args.transformer_dropout}"
        name += f"-gdrop={args.gnn_dropout}"
        name += "-pretrained_gnn" if args.pretrained_gnn else ""
        name += f"-freeze_gnn={args.freeze_gnn}" if args.freeze_gnn is not None else ""
        return name

    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args):
        super().__init__()
        self.gnn_node = PNANodeEmbedding(node_encoder, args)
        if args.pretrained_gnn:
            # logger.info(self.gnn_node)
            state_dict = torch.load(args.pretrained_gnn)
            state_dict = self._gnn_node_state(state_dict["model"])
            logger.info("Load GNN state from: {}", state_dict.keys())
            self.gnn_node.load_state_dict(state_dict)
        self.freeze_gnn = args.freeze_gnn

        gnn_emb_dim = 2 * args.gnn_emb_dim if args.gnn_JK == "cat" else args.gnn_emb_dim
        self.gnn2transformer = nn.Linear(gnn_emb_dim, args.d_model)
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
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]

        padded_h_node, src_padding_mask = pad_batch(h_node, batched_data.batch, self.transformer_encoder.max_input_len)  # Pad in the front

        transformer_out, mask = self.transformer_encoder(padded_h_node, src_padding_mask)  # [s, b, h], [b, s]

        if self.pooling in ["last", "cls"]:
            h_graph = transformer_out[-1]
        elif self.pooling == "mean":
            h_graph = transformer_out.sum(0) / (~mask).sum(-1, keepdim=True)
        else:
            raise NotImplementedError

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
