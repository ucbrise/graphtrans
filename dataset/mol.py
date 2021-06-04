import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from tqdm import tqdm


class MolUtil:
    @staticmethod
    def add_args(parser):
        parser.add_argument("--feature", type=str, default="full", help="full feature or simple feature")
        parser.set_defaults(batch_size=32)
        parser.set_defaults(epochs=100)
        parser.set_defaults(gnn_dropout=0.5)

    @staticmethod
    def loss_fn(task_type):
        cls_criterion = torch.nn.BCEWithLogitsLoss()
        reg_criterion = torch.nn.MSELoss()

        def calc_loss(pred, batch, m=1.0):
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
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

                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return evaluator.eval(input_dict)

    @staticmethod
    def preprocess(dataset, dataset_eval, model_cls, args):
        split_idx = dataset.get_idx_split()
        if args.feature == "full":
            pass
        elif args.feature == "simple":
            logger.debug("using simple feature")
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
        # Compute in-degree histogram over training data.
        deg = torch.zeros(10, dtype=torch.long)
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

        node_encoder_cls = lambda: AtomEncoder(model_cls.get_emb_dim(args))
        edge_encoder_cls = lambda emb_dim: BondEncoder(emb_dim=emb_dim)
        return dataset.num_tasks, node_encoder_cls, edge_encoder_cls, deg
