from models.gat_transformer import GATTransformer
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models.gat import GAT
from models.gat_transformer import GATTransformer
from ray import tune
import numpy as np



dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

def train(config):
    acc_result = []
    for _ in range(5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(device)

        # model, data = GAT(dataset).to(device), data.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
        data = dataset[0]
        model, data = GATTransformer(dataset, config).to(device), data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = None
        if config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        def train():
            model.train()
            optimizer.zero_grad()
            F.cross_entropy(model(data)[data.train_mask], data.y[data.train_mask]).backward()
            optimizer.step()

        def test():
            model.eval()
            log_probs, accs = model(data), []
            for _, mask in data('train_mask', 'test_mask'):
                pred = log_probs[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            return accs

        for epoch in range(1, 201):
            train()
            if scheduler is not None: scheduler.step()
            log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
            result = test()
            train_acc, test_acc = result
            # tune.report(train_acc=train_acc, test_acc=test_acc)
            # print(log.format(epoch, *result))
        acc_result.append(test_acc)
        print(test_acc)
        if test_acc < 0.7:
            break
    print("mean_acc: ", np.mean(acc_result))
    return np.mean(acc_result)

def training_function(config):
    result = train(config)
    tune.report(mean_acc=result)
    return result


if __name__ == "__main__":
    # training_function(
    #     config={'dp_1': 0.8, 'dp_2': 0.6, 'gat_dp': 0.6, 'gat_out_dim': 32, 'lr': 0.001, 'trans_dp': 0.6, 'trans_expand': 4, 'trans_head': 2, 'trans_nlayer': 4, 'wd': 0.005, "JK": "cat"})

    analysis = tune.run(
        training_function,
        num_samples=20000,
        config={
            "lr": tune.choice([0.0001, 0.001, 0.01, 0.1]),
            "wd": tune.choice([5e-4, 1e-4, 1e-3, 5e-3]),
            "gat_out_dim": tune.choice([8, 16, 32, 64]),
            "gat_dp": tune.choice([0.3, 0.4, 0.5, 0.6, 0.8]),
            "trans_dp": tune.choice([0.3, 0.4, 0.5, 0.6, 0.8]),
            "dp_1": tune.choice([0.3, 0.4, 0.5, 0.6, 0.8]),
            "dp_2": tune.choice([0.3, 0.4, 0.5, 0.6, 0.8]),
            "trans_nlayer": tune.choice([1, 2, 3, 4]),
            "trans_head": tune.choice([1, 2, 4]),
            "trans_expand": tune.choice([1, 2, 4]),
            "JK": tune.choice(["cat"]),
            "double_linear": tune.choice([True]),
        },
        resources_per_trial={"gpu": 0.3},
        local_dir="/data/zhwu/gat/ray_results")

    # print("best config: ", analysis.get_best_config(metric="test_acc", mode="max"))
