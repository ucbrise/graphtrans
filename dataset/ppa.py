import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np
from torchvision import transforms
from warnings import warn


class PPAUtil:
    def __init__(self):
        warn("This PPA method has not been tested yet.")

    @staticmethod
    def add_args(parser):
        parser.set_defaults(gnn_dropout=0.5)
        parser.set_defaults(batch_size=32)
        parser.set_defaults(epochs=100)

    @staticmethod
    def loss_fn(_):
        multicls_criterion = torch.nn.CrossEntropyLoss()

        def calc_loss(pred, batch, m=1.0):
            loss = multicls_criterion(
                pred.to(torch.float32),
                batch.y.view(
                    -1,
                ),
            )
            loss /= m
            return loss

        return calc_loss

    @staticmethod
    def eval(model, device, loader, evaluator):
        model.eval()
        y_true = []
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = batch.to(device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)

                y_true.append(batch.y.view(-1, 1).detach().cpu())
                y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return evaluator.eval(input_dict)

    @staticmethod
    def preprocess(dataset, dataset_eval, model_cls, args):
        split_idx = dataset.get_idx_split()
        dataset_transform = [add_zeros]
        dataset_eval.transform = transforms.Compose(dataset_transform)
        if dataset.transform is not None:
            dataset_transform.append(dataset.transform)
        dataset.transform = transforms.Compose(dataset_transform)
        edge_encoder_cls = lambda emb_dim: nn.Linear(7, emb_dim)
        node_encoder_cls = lambda: nn.Embedding(1, args.gnn_emb_dim)
        return dataset.num_classes, node_encoder_cls, edge_encoder_cls, None


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data
