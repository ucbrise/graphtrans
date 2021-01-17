import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.utils import degree

from tqdm import tqdm
import configargparse
import time
import numpy as np
import pandas as pd
import os

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from trainers import get_trainer_and_parser
from models import get_model_and_parser
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
wandb.init(project='graph-aug')

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def main():
    # fmt: off
    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                    description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)

    parser.add_argument('--data_root', type=str, default='/data/zhwu/ogb')
    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag]')
                        
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
    parser.add_argument('--scheduler', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--test-freq', type=int, default=1)
    # fmt: on

    args, _ = parser.parse_known_args()
    
    # Setup Trainer and add customized args
    trainer = get_trainer_and_parser(args, parser)
    train = trainer.train
    model_cls = get_model_and_parser(args, parser)

    args = parser.parse_args()

    run_name = f'{args.dataset}+{args.gnn}+{args.aug}'
    wandb.run.name = run_name
    wandb.run.save()

    device = torch.device("cuda") if torch.cuda.is_available() and args.device >= 0 else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset, root=args.data_root)
    task_type = dataset.task_type

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # Compute in-degree histogram over training data.
    deg = torch.zeros(10, dtype=torch.long)
    for data in dataset[split_idx['train']]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    args.deg = deg


    def calc_loss(pred, batch, m=1.0):
        is_labeled = batch.y == batch.y
        if "classification" in task_type: 
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        else:
            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        loss /= m
        return loss

    def run(run_id):
        best_val, final_test = 0, 0
        model = model_cls(num_tasks=dataset.num_tasks, args=args, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.0001)

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}=====".format(epoch))
            print('Training...')
            loss = train(model, device, train_loader, optimizer, args, calc_loss)

            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_metric, valid_metric, test_metric = train_perf[dataset.eval_metric], valid_perf[dataset.eval_metric], test_perf[dataset.eval_metric]
            wandb.log({f'train/{dataset.eval_metric}-runs{run_id}': train_metric,
                    f'valid/{dataset.eval_metric}-runs{run_id}': valid_metric,
                    f'test/{dataset.eval_metric}-runs{run_id}': test_metric,
                    'epoch': epoch})
            print(f"Running: {run_name} (runs {run_id})")
            print(f"Run {run_id} - train: {train_metric}, val: {valid_metric}, test: {test_metric}")

            if args.scheduler:
               scheduler.step(valid_metric)

            if best_val < valid_metric:
                best_val = valid_metric
                final_test = test_metric
        return best_val, final_test

    vals, tests = [], []
    for run_id in range(args.runs):
        best_val, final_test = run(run_id)
        vals.append(best_val)
        tests.append(final_test)
        print(f'Run {run} - val: {best_val}, test: {final_test}')
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")

if __name__ == "__main__":
    main()
