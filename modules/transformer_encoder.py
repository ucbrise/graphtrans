import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.gnn as gnn


class TransformerNodeEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("transformer")
        group.add_argument("--d_model", type=int, default=128, help="transformer d_model.")
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
        group.add_argument("--transformer_dropout", type=float, default=0.3)
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--num_encoder_layers", type=int, default=4)
        group.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
        group.add_argument("--transformer_norm_input", action="store_true", default=False)

    def __init__(self, args):
        super().__init__()

        self.d_model = args.d_model
        self.num_layer = args.num_encoder_layers
        # Creating Transformer Encoder Model
        encoder_layer = nn.TransformerEncoderLayer(
            args.d_model, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation
        )
        encoder_norm = nn.LayerNorm(args.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, args.num_encoder_layers, encoder_norm)
        self.max_input_len = args.max_input_len

        self.norm_input = None
        if args.transformer_norm_input:
            self.norm_input = nn.LayerNorm(args.d_model)
        self.cls_embedding = None
        if args.graph_pooling == "cls":
            self.cls_embedding = nn.Parameter(torch.randn([1, 1, args.d_model], requires_grad=True))

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """

        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)

            zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask)  # (S, B, h_d)

        return transformer_out, src_padding_mask
