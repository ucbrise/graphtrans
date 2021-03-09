import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.utils import degree
from data.encoders import EDGE_ENCODERS
from data import LOSSES, EVALS
import random
from tqdm import tqdm
import configargparse
import time
import numpy as np
import pandas as pd
import os

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# importing utils
from utils import ASTNodeEncoder, get_vocab_mapping
# for data transform
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq

from trainers import get_trainer_and_parser
from models import get_model_and_parser
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime
import wandb
wandb.init(project='graph-aug')
now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")

def main():
    # fmt: off
    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                    description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)


    parser.add_argument('--data_root', type=str, default='/data/zhwu/ogb')
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag]')
                        
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    group = parser.add_argument_group('model')
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
    group.add_argument('--devices', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    group.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    group.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    group.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    group.add_argument('--scheduler', type=bool, default=False)
    group.add_argument('--weight_decay', type=float, default=0.0)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--test-freq', type=int, default=1)
    group.add_argument('--resume', type=str, default=None)
    group.add_argument('--seed', type=int, default=None)
    # fmt: on

    args, _ = parser.parse_known_args()
    
    # Setup Trainer and add customized args
    trainer = get_trainer_and_parser(args, parser)
    train = trainer.train
    model_cls = get_model_and_parser(args, parser)
    args = parser.parse_args()
    data_transform = trainer.transform(args)

    run_name = f'{args.dataset}+{args.gnn_type}'
    run_name += '-virtual' if args.gnn_virtual_node else ''
    run_name += f'+{trainer.name(args)}'
    if args.scheduler:
        run_name = run_name+f'+scheduler'
    if args.seed:
        run_name = run_name + f'+seed{args.seed}'
    wandb.run.name = run_name
    wandb.run.save()

    device = torch.device("cuda") if torch.cuda.is_available() and args.devices >= 0 else torch.device("cpu")
    args.save_path = f'exps/{run_name}-{now}'
    os.makedirs(args.save_path, exist_ok=True)
    if args.resume is not None:
        args.save_path = args.resume
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if device == torch.cuda.is_available():
            cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)

    # automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, root=args.data_root, transform=data_transform)
    dataset_eval = PygGraphPropPredDataset(name = args.dataset, root=args.data_root)
    task_type = dataset.task_type
    
    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(
        args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list)))

    split_idx = dataset.get_idx_split()

    # building vocabulary for sequence predition. Only use training data.
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    # set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset_transform = [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)]
    dataset_eval.transform = transforms.Compose(dataset_transform)
    if dataset.transform is not None:
        dataset_transform.append(dataset.transform)
    dataset.transform = transforms.Compose(dataset_transform)

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset_eval[split_idx["valid"]], batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    # Encoding node features into gnn_emb_dim vectors.
    # The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(args.gnn_emb_dim, num_nodetypes=len(
        nodetypes_mapping['type']), num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)
    edge_encoder_cls = EDGE_ENCODERS[args.dataset]

    # Compute in-degree histogram over training data.
    if model_cls.need_deg():
        deg = torch.zeros(800, dtype=torch.long)
        num_nodes = 0.0
        num_graphs = 0
        for data in dataset_eval[split_idx['train']]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
            num_nodes += data.num_nodes
            num_graphs +=1
        args.deg = deg
        print("Avg num nodes:", num_nodes / num_graphs)
        print("Avg deg:", deg)

    calc_loss = LOSSES[args.dataset](task_type)
    eval = EVALS[args.dataset]

    def run(run_id):
        os.makedirs(os.path.join(args.save_path, str(run_id)), exist_ok=True)
        best_val, final_test = 0, 0
        model = model_cls(num_tasks=len(vocab2idx), args=args, node_encoder=node_encoder, edge_encoder_cls=edge_encoder_cls).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.0001, verbose=True)

        # Load resume model, if any
        start_epoch = 1
        last_model_path = os.path.join(args.save_path, str(run_id), 'last_model.pt')
        if os.path.exists(last_model_path):
            state_dict = torch.load(last_model_path)
            start_epoch = state_dict["epoch"] + 1
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            if args.scheduler:
                scheduler.load_state_dict(state_dict['scheduler'])
            print("[Resume] Loaded:", last_model_path, "epoch:", start_epoch)


        for epoch in range(start_epoch, args.epochs + 1):
            print("=====Epoch {}=====".format(epoch))
            print('Training...')
            loss = train(model, device, train_loader, optimizer, args, calc_loss)

            if args.scheduler:
                valid_perf = eval(model, device, valid_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
                valid_metric = valid_perf[dataset.eval_metric]
                scheduler.step(valid_metric)
            if epoch > args.epochs // 2 and epoch % args.test_freq == 0 or epoch in [1, args.epochs]:
                print('Evaluating...')
                train_perf = eval(model, device, train_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
                valid_perf = eval(model, device, valid_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
                test_perf = eval(model, device, test_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

                # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

                train_metric, valid_metric, test_metric = train_perf[dataset.eval_metric], valid_perf[dataset.eval_metric], test_perf[dataset.eval_metric]
                wandb.log({f'train/{dataset.eval_metric}-runs{run_id}': train_metric,
                        f'valid/{dataset.eval_metric}-runs{run_id}': valid_metric,
                        f'test/{dataset.eval_metric}-runs{run_id}': test_metric,
                        'epoch': epoch})
                print(f"Running: {run_name} (runs {run_id})")
                print(f"Run {run_id} - train: {train_metric}, val: {valid_metric}, test: {test_metric}")
                
                # Save checkpoints
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }
                state_dict["scheduler"] = scheduler.state_dict() if args.scheduler else None
                torch.save(state_dict, os.path.join(args.save_path, str(run_id), 'last_model.pt'))
                print("[Save] Save model:", os.path.join(args.save_path, str(run_id), 'last_model.pt'))
                if best_val < valid_metric:
                    best_val = valid_metric
                    final_test = test_metric
                    torch.save(state_dict, os.path.join(args.save_path, str(run_id), 'best_model.pt'))
                    print("[Best Model] Save model:", os.path.join(args.save_path, str(run_id), 'best_model.pt'))
                    
        state_dict = torch.load(os.path.join(args.save_path, str(run_id), 'best_model.pt'))
        print("[Evaluate] Loaded from", os.path.join(args.save_path, str(run_id), 'best_model.pt'))
        model.load_state_dict(state_dict['model'])
        best_valid_perf = eval(model, device, valid_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        best_test_perf = eval(model, device, test_loader, evaluator,
                              arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        return best_valid_perf[dataset.eval_metric], best_test_perf[dataset.eval_metric]

    vals, tests = [], []
    for run_id in range(args.runs):
        best_val, final_test = run(run_id)
        vals.append(best_val)
        tests.append(final_test)
        print(f'Run {run_id} - val: {best_val}, test: {final_test}')
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")

if __name__ == "__main__":
    main()