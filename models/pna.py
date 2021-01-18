import torch.nn as nn
from modules.pna_layer import PNAConvSimple
import torch
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn.functional as F


class PNANet(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('PNANet configs')
        group.add_argument('--aggregators', type=str, nargs='+', default=['mean', 'max', 'min', 'std'])
        group.add_argument('--scalers', type=str, nargs='+', default=['identity', 'amplification', 'attenuation'])
        group.add_argument('--post_layers', type=int, default=1)
        
    def __init__(self, num_tasks, node_encoder, edge_encoder_cls, args, max_seq_len=None, add_edge='none', num_layer=4, emb_dim=70, out_dim=70, graph_pooling="mean", residual=True, drop_ratio=0.3):
        super().__init__()
        self.num_layer = num_layer
        self.num_tasks = num_tasks
        self.max_seq_len = max_seq_len
        self.aggregators = args.aggregators
        self.scalers = args.scalers
        self.residual = residual
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling

        self.node_encoder = node_encoder

        self.layers = nn.ModuleList(
            [PNAConvSimple(edge_encoder_cls=edge_encoder_cls, add_edge=add_edge, in_channels=emb_dim, out_channels=emb_dim, aggregators=self.aggregators, scalers=self.scalers, deg=args.deg, post_layers=args.post_layers, drop_ratio=drop_ratio)
             for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([BatchNorm(emb_dim) for _ in range(num_layer)])

        if self.max_seq_len is None:
            self.mlp = nn.Sequential(nn.Linear(emb_dim, 35, bias=True), 
                        nn.ReLU(), 
                        nn.Linear(35, 17, bias=True),
                        nn.ReLU(), 
                        nn.Linear(17, num_tasks, bias=True))

        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.ReLU(), 
                    nn.Linear(emb_dim, self.num_tasks),
                    ))

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None
        encoded_node = self.node_encoder(x) if node_depth is None else self.node_encoder(x, node_depth.view(-1,))
        x = encoded_node + perturb if perturb is not None else encoded_node

        for conv, batch_norm in zip(self.layers, self.batch_norms):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            if self.residual:
                x = h + x
            x = F.dropout(x, self.drop_ratio, training=self.training)

        h_graph = self.pool(x, batched_data.batch)

        if self.max_seq_len is None:
            return self.mlp(h_graph)
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))
        return pred_list
