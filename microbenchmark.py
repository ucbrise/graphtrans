import os
import random
from datetime import datetime
from tqdm import tqdm

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from loguru import logger
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch_geometric.data import DataLoader
from tqdm import tqdm

import utils
from data.adj_list import compute_adjacency_list_cached
from dataset import DATASET_UTILS
from models import get_model_and_parser
from trainers import get_trainer_and_parser
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.data import Batch
import time

wandb.init(project="graph-aug")
now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")


def main():
    # fmt: off
    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                    description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)
    parser.add_argument('--wandb_run_idx', type=str, default=None)


    parser.add_argument('--data_root', type=str, default='/data/zhwu/ogb')
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag|augment]')
                        
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='maximum sequence length to predict (default: None)')
    parser.add_argument('--num_nodes', type=int, default=100)
   
    group = parser.add_argument_group('model')
    group.add_argument('--model_type', type=str, default='gnn', help='gnn|pna|gnn-transformer')
    group.add_argument('--graph_pooling', type=str, default='mean')
    group = parser.add_argument_group('gnn')
    group.add_argument('--gnn_type', type=str, default='gcn')
    group.add_argument('--gnn_virtual_node', action='store_true')
    group.add_argument('--gnn_dropout', type=float, default=0)
    group.add_argument('--gnn_num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    group.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    group.add_argument('--gnn_JK', type=str, default='last')
    group.add_argument('--gnn_residual', action='store_true', default=False)

    group = parser.add_argument_group('training')
    group.add_argument('--devices', type=str, default="0",
                        help='which gpu to use if any (default: 0)')
    group.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    group.add_argument('--eval_batch_size', type=int, default=None,
                        help='input batch size for training (default: train batch size)')
    group.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    group.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    group.add_argument('--scheduler', type=str, default=None)
    group.add_argument('--pct_start', type=float, default=0.3)
    group.add_argument('--weight_decay', type=float, default=0.0)
    group.add_argument('--grad_clip', type=float, default=None)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--max_lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--test-freq', type=int, default=1)
    group.add_argument('--start-eval', type=int, default=15)
    group.add_argument('--resume', type=str, default=None)
    group.add_argument('--seed', type=int, default=None)
    # fmt: on

    args, _ = parser.parse_known_args()
    dataset_util = DATASET_UTILS[args.dataset]()
    dataset_util.add_args(parser)
    args, _ = parser.parse_known_args()

    # Setup Trainer and add customized args
    trainer = get_trainer_and_parser(args, parser)
    train = trainer.train
    model_cls = get_model_and_parser(args, parser)
    args = parser.parse_args()
    data_transform = trainer.transform(args)
    num_tasks = 10
    loss_fn = dataset_util.loss_fn(None)
    print("max_seq_len:", args.max_seq_len)

    run_name = f"{args.dataset}+{model_cls.name(args)}"
    run_name += f"+{trainer.name(args)}+lr={args.lr}+wd={args.weight_decay}"
    if args.scheduler is not None:
        run_name = run_name + f"+sch={args.scheduler}"
    if args.seed:
        run_name = run_name + f"+seed{args.seed}"
    if args.wandb_run_idx is not None:
        run_name = args.wandb_run_idx + "_" + run_name

    wandb.run.name = run_name

    device = torch.device("cuda") if torch.cuda.is_available() and args.devices else torch.device("cpu")
    args.save_path = f"exps/{run_name}-{now}"
    os.makedirs(args.save_path, exist_ok=True)
    if args.resume is not None:
        args.save_path = args.resume
    logger.info(args)
    wandb.config.update(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if device == torch.cuda.is_available():
            # cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)


    forward_time = []
    backward_time = []
    model = model_cls(num_tasks=num_tasks, args=args, node_encoder=nn.Identity(), edge_encoder_cls=lambda dim: nn.Linear(2, dim)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    edge_prob_list = [prob / 10 for prob in range(2, 10, 2)]
    model.train()
    for edge_prob in edge_prob_list:
        timers = {name: CUDAWindowedTimer() for name in ["forward", "backward"]}
        print("edge_prob", edge_prob)
        try:
            for _ in range(100):
                model.zero_grad()
                edge_index = torch.cat([erdos_renyi_graph(args.num_nodes, edge_prob).to(device) for _ in range(args.batch_size)], dim=1)
                features = torch.randn(args.num_nodes * args.batch_size, args.gnn_emb_dim).to(device)
                label = torch.ones((args.batch_size, args.max_seq_len), dtype=torch.long).to(device)

                batch = torch.zeros(args.num_nodes * args.batch_size, dtype=torch.long, device=device)
                for i in range(args.batch_size):
                    batch[i * args.num_nodes: (i+1) * args.num_nodes] = i
                batch = Batch(batch=batch, edge_index=edge_index, x=features, y_arr=label, edge_attr=torch.randn((edge_index.size(1), 2), device=device))

                with timers['forward']:
                    y = model(batch)
                    # print(y.size())

                with timers['backward']:
                    loss = loss_fn(y, batch)
                    loss.backward()
                    optimizer.step()
            cur_forward_time = timers['forward'].value() * 1000
            cur_backward_time = timers['backward'].value() * 1000
            forward_time.append(f'{cur_forward_time:.2f}')
            backward_time.append(f'{cur_backward_time:.2f}')
        except:
            forward_time.append(f'OOM')
            backward_time.append(f'OOM')
    print('\t'.join([str(prob) for prob in edge_prob_list]))
    print('\t'.join(forward_time))
    print('\t'.join(backward_time))



class WindowedTimer():
    def __init__(self, window_size=100, timer_type='CPU'):
        self.timer = 0
        self.window = [0] * window_size
        self.window_size = window_size
        self.ind = 0
        self.total_cnt = 0

        self.sum = 0.0
        self.milestone = 0.0

        self.timer_type = timer_type

    def start(self):
        self.timer = time.time()
        if self.timer_type == 'GPU':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()

    def end(self):
        if self.timer_type == 'GPU':
            self.end_event.record()
            torch.cuda.synchronize()
            self.window[self.ind] = self.start_event.elapsed_time(self.end_event) / 1000
        else:
            self.window[self.ind] = time.time() - self.timer
        self.sum += self.window[self.ind]
        self.milestone += self.window[self.ind]
        self.ind = (self.ind + 1) % self.window_size
        self.total_cnt = min(self.total_cnt + 1, self.window_size)

    def __enter__(self):
        self.start()

    def __exit__(self, *exec):
        self.end()

    def last(self):
        return self.window[self.ind - 1]

    def value(self):
        if self.total_cnt == 0: return 0
        return sum(self.window) / self.total_cnt

    def start_milestone(self):
        self.milestone = 0


def CUDAWindowedTimer(window_size=100):
    return WindowedTimer(window_size, timer_type='GPU')


if __name__ == "__main__":
    main()
