import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from tqdm import tqdm


class TUUtil:
    @staticmethod
    def add_args(parser):
        parser.set_defaults(batch_size=128)
        parser.set_defaults(epochs=10000)
        parser.set_defaults(lr=0.0005)
        parser.set_defaults(weight_decay=0.0001)
        parser.set_defaults(gnn_dropout=0.5)
        parser.set_defaults(gnn_emb_dim=128)

    @staticmethod
    def loss_fn(task_type):
        def calc_loss(pred, batch, m=1.0):
            loss = F.cross_entropy(pred, batch.y)
            return loss

        return calc_loss

    @staticmethod
    @torch.no_grad()
    def eval(model, device, loader, evaluator):
        model.eval()

        correct = 0
        for step, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = batch.to(device)

            pred = model(batch)
            pred = pred.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()
        return {"acc": correct / len(loader.dataset)}

    @staticmethod
    def preprocess(args):
        dataset = TUDataset(os.path.join(args.data_root, args.dataset), name=args.dataset)
        num_tasks = dataset.num_classes

        num_features = dataset.num_features

        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        class Dataset(dict):
            pass

        dataset = Dataset({"train": training_set, "valid": validation_set, "test": test_set})
        dataset.eval_metric = "acc"
        dataset.task_type = "classification"
        dataset.get_idx_split = lambda: {"train": "train", "valid": "valid", "test": "test"}

        node_encoder_cls = lambda: nn.Linear(num_features, args.gnn_emb_dim)

        def edge_encoder_cls(_):
            def zero(_):
                return 0

            return zero

        return dataset, num_tasks, node_encoder_cls, edge_encoder_cls, None
