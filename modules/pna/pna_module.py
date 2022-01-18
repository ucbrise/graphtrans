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


class PNANodeEmbedding(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("PNANet configs")
        group.add_argument("--aggregators", type=str, nargs="+", default=["mean", "max", "min", "std"])
        group.add_argument("--scalers", type=str, nargs="+", default=["identity", "amplification", "attenuation"])
        group.add_argument("--post_layers", type=int, default=1)
        group.add_argument("--add_edge", type=str, default="none")
        group.set_defaults(gnn_residual=True)
        group.set_defaults(gnn_dropout=0.3)
        group.set_defaults(gnn_emb_dim=70)
        group.set_defaults(gnn_num_layer=4)

    def __init__(self, node_encoder, args):
        super().__init__()
        self.num_layer = args.gnn_num_layer
        self.max_seq_len = args.max_seq_len
        self.aggregators = args.aggregators
        self.scalers = args.scalers
        self.residual = args.gnn_residual
        self.drop_ratio = args.gnn_dropout
        self.graph_pooling = args.graph_pooling

        self.node_encoder = node_encoder

        self.layers = nn.ModuleList(
            [
                PNAConv(
                    args.gnn_emb_dim,
                    args.gnn_emb_dim,
                    aggregators=self.aggregators,
                    scalers=self.scalers,
                    deg=args.deg,
                    towers=4,
                    divide_input=True,
                )
                for _ in range(self.num_layer)
            ]
        )
        self.batch_norms = nn.ModuleList([BatchNorm(args.gnn_emb_dim) for _ in range(self.num_layer)])

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
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
        x = encoded_node + perturb if perturb is not None else encoded_node

        for conv, batch_norm in zip(self.layers, self.batch_norms):
            h = F.relu(batch_norm(conv(x, edge_index)))
            if self.residual:
                x = h + x
            x = F.dropout(x, self.drop_ratio, training=self.training)

        return x
