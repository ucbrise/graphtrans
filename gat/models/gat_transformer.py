import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .gat import GAT
from .transformer_encoder import TransformerNodeEncoder

class GATTransformer(torch.nn.Module):
    def __init__(self, dataset, config):
        super().__init__()
        out_dim = config["gat_out_dim"]
        self.gat = GAT(dataset, out_dim=out_dim, dropout=config["gat_dp"], JK=config["JK"], double_linear=config['double_linear'])
        if config["JK"] == 'cat':
            out_dim *= 2
        class Args:
            def __init__(self):
                self.d_model = out_dim
                self.dim_feedforward = config["trans_expand"] * out_dim
                self.transformer_dropout = config["trans_dp"]
                self.transformer_norm_input = True

                # default
                self.nhead = config["trans_head"]
                self.transformer_activation = 'relu'
                self.graph_pooling = None
                self.num_encoder_layers = config["trans_nlayer"]
                self.max_input_len = 10000

        self.dp_1 = config["dp_1"]
        self.dp_2 = config["dp_2"]

        self.transformer_encoder = TransformerNodeEncoder(Args())
        self.out_linear = torch.nn.Linear(out_dim, dataset.num_classes)
        
    def forward(self, data):
        x = self.gat(data)
        x = F.elu(x)
        x = F.dropout(x, self.dp_1, training=self.training)
        padded_x, padding_mask = pad_batch(x, torch.zeros(len(x), dtype=torch.int), max_input_len=10000)
        x, _ = self.transformer_encoder(padded_x, padding_mask)
        x = x.squeeze(1)
        x = F.dropout(x, self.dp_2, training=self.training)
        x = self.out_linear(x)
        # print("out dim", x.size())
        return x


def pad_batch(h_node, batch, max_input_len, get_mask=False):
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    # logger.info(max(num_nodes))
    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask
