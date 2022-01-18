import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torchvision import transforms
from tqdm import tqdm

from loguru import logger
# for data transform
# importing utils
from .utils import (
    ASTNodeEncoder,
    augment_edge,
    decode_arr_to_seq,
    encode_y_to_arr,
    get_vocab_mapping,
)


class CodeUtil:
    def __init__(self):
        self.arr_to_seq = None

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--num_vocab", type=int, default=5000, help="the number of vocabulary used for sequence prediction (default: 5000)"
        )
        parser.set_defaults(max_seq_len=5)

    @staticmethod
    def loss_fn(_):
        multicls_criterion = torch.nn.CrossEntropyLoss()

        def calc_loss(pred_list, batch, m=1.0):
            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])
            loss = loss / len(pred_list)
            loss /= m
            return loss

        return calc_loss

    def eval(self, model, device, loader, evaluator):
        model.eval()
        seq_ref_list = []
        seq_pred_list = []

        for step, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = batch.to(device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred_list = model(batch)

                mat = []
                for i in range(len(pred_list)):
                    mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
                mat = torch.cat(mat, dim=1)

                seq_pred = [self.arr_to_seq(arr) for arr in mat]

                # PyG >= 1.5.0
                seq_ref = [batch.y[i] for i in range(len(batch.y))]

                seq_ref_list.extend(seq_ref)
                seq_pred_list.extend(seq_pred)

        input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

        return evaluator.eval(input_dict)

    def preprocess(self, dataset, dataset_eval, model_cls, args):
        split_idx = dataset.get_idx_split()
        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        print(
            "Target seqence less or equal to {} is {}%.".format(
                args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list)
            )
        )

        # building vocabulary for sequence predition. Only use training data.
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx["train"]], args.num_vocab)

        self.arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)

        # set the transform function
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
        dataset_transform = [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)]
        dataset_eval.transform = transforms.Compose(dataset_transform)
        if dataset.transform is not None:
            dataset_transform.append(dataset.transform)
        dataset.transform = transforms.Compose(dataset_transform)

        nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, "mapping", "typeidx2type.csv.gz"))
        nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, "mapping", "attridx2attr.csv.gz"))

        # Encoding node features into gnn_emb_dim vectors.
        # The following three node features are used.
        # 1. node type
        # 2. node attribute
        # 3. node depth
        node_encoder_cls = lambda: ASTNodeEncoder(
            args.gnn_emb_dim,
            num_nodetypes=len(nodetypes_mapping["type"]),
            num_nodeattributes=len(nodeattributes_mapping["attr"]),
            max_depth=20,
        )
        edge_encoder_cls = lambda emb_dim: nn.Linear(2, emb_dim)

        deg = None
        # Compute in-degree histogram over training data.
        if model_cls.need_deg():
            deg = torch.zeros(800, dtype=torch.long)
            num_nodes = 0.0
            num_graphs = 0
            for data in dataset_eval[split_idx["train"]]:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                deg += torch.bincount(d, minlength=deg.numel())
                num_nodes += data.num_nodes
                num_graphs += 1
            args.deg = deg
            logger.debug("Avg num nodes: {}", num_nodes / num_graphs)
            logger.debug("Avg deg: {}", deg)
        return len(vocab2idx), node_encoder_cls, edge_encoder_cls, deg
