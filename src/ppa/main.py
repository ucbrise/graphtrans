import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn import GNN

from tqdm import tqdm
import configargparse
import time
import numpy as np
import pandas as pd
import os

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from trainers import get_trainer

import wandb
wandb.init(project='graph-aug')

multicls_criterion = torch.nn.CrossEntropyLoss()
def calc_loss(pred, batch, m=1.0):
    loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
    loss /= m
    return loss

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

            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

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
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--test-freq', type=int, default=1)
    # fmt: on

    args, _ = parser.parse_known_args()
    
    # Setup Trainer and add customized args
    trainer = get_trainer(args)
    trainer.add_args(parser)
    train = trainer.train

    args = parser.parse_args()

    run_name = f'{args.dataset}+{args.gnn}+{args.aug}'
    wandb.run.name = run_name
    wandb.run.save()

    device = torch.device("cuda") if torch.cuda.is_available() and args.device >= 0 else torch.device("cpu")

    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros, root=args.data_root)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    def run(run_id):
        best_val, final_test = 0, 0
        if args.gnn == 'gin':
            model = GNN(gnn_type='gin', num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=False).to(device)
        elif args.gnn == 'gin-virtual':
            model = GNN(gnn_type='gin', num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        elif args.gnn == 'gcn':
            model = GNN(gnn_type='gcn', num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=False).to(device)
        elif args.gnn == 'gcn-virtual':
            model = GNN(gnn_type='gcn', num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        else:
            raise ValueError('Invalid GNN type')


        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}=====".format(epoch))
            print('Training...')
            loss = train(model, device, train_loader, optimizer, args, calc_loss)

            if epoch > args.epochs // 2 and epoch % args.test_freq == 0 or epoch in [1, args.epochs]:
                print('Evaluating...')
                train_perf = eval(model, device, train_loader, evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator)
                test_perf = eval(model, device, test_loader, evaluator)

                # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

                train_metric, valid_metric, test_metric = train_perf[dataset.eval_metric], valid_perf[dataset.eval_metric], test_perf[dataset.eval_metric]
                wandb.log({f'train/{dataset.eval_metric}': train_metric,
                        f'valid/{dataset.eval_metric}': valid_metric,
                        f'test/{dataset.eval_metric}-runs{run_id}': test_metric,
                        'epoch': epoch})

                if best_val > valid_metric:
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
