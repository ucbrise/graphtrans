import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import pad_batch, unpad_batch

class MultiBranchNode(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group('multibranch')
        group.add_argument('--multibranch_residual', action='store_true', default=False)

    def __init__(self, gnn_node, transformer, args):
        super().__init__()
        assert gnn_node.num_layer == transformer.num_layer
        self.gnn_node = gnn_node

        self.transformer_layers = transformer.transformer.layers
        self.num_layer = transformer.num_layer
        self.max_input_len = transformer.max_input_len
        self.residual = args.multibranch_residual

        self.node_encoder = self.gnn_node.node_encoder

        self.emb_dim = args.gnn_emb_dim + args.d_model
        self.fuse_gnn = nn.ModuleList([nn.Linear(self.emb_dim, args.gnn_emb_dim) for _ in range(self.num_layer)])
        self.fuse_tran = nn.ModuleList([nn.Linear(self.emb_dim, args.d_model) for _ in range(self.num_layer)])

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None
        encoded_node = self.node_encoder(x) if node_depth is None else self.node_encoder(x, node_depth.view(-1, ))
        tmp = encoded_node + perturb if perturb is not None else encoded_node
        h_list = [tmp]
        
        for layer in range(self.num_layer):
            gnn_in = self.fuse_gnn[layer](h_list[layer])
            tran_in = self.fuse_tran[layer](h_list[layer])
            h_gnn = self.gnn_node.convs[layer](gnn_in, edge_index, edge_attr)
            padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes = pad_batch(tran_in, batch, self.max_input_len, get_mask=True)
            h_tran = self.transformer_layers[layer](padded_h_node, src_key_padding_mask=src_padding_mask)

            h_tran = unpad_batch(h_tran, tran_in, num_nodes, masks, max_num_nodes)
            tmp = torch.cat([h_gnn, h_tran], dim=-1)
            
            if self.residual:
                tmp = tmp + h_list[layer]
            h_list.append(tmp)

        return h_list[-1]

