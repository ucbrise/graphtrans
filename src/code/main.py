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

# importing utils
from utils import ASTNodeEncoder, get_vocab_mapping
# for data transform
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq

import sys
from trainers import get_trainer_and_parser
from models import get_model_and_parser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.encoders import EDGE_ENCODERS

from datetime import datetime
import wandb
wandb.init(project='graph-aug')
now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")

multicls_criterion = torch.nn.CrossEntropyLoss()
def calc_loss(pred_list, batch, m=1.0):
    loss = 0
    for i in range(len(pred_list)):
        loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])
    loss = loss / len(pred_list)
    loss /= m
    return loss

def eval(model, device, loader, evaluator, arr_to_seq):
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

            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

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
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')
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
    data_transform = trainer.transform(args)

    run_name = f'{args.dataset}+{args.gnn}+{trainer.name(args)}'
    if args.scheduler:
        run_name = run_name+f'+scheduler'
    wandb.run.name = run_name
    wandb.run.save()

    device = torch.device("cuda") if torch.cuda.is_available() and args.device >= 0 else torch.device("cpu")
    args.save_path = f'exps/{run_name}-{now}'
    os.makedirs(args.save_path, exist_ok=True)
    if args.resume is not None:
        args.save_path = args.resume
    print(args)

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
    if dataset.transform is None:
        dataset.transform = transforms.Compose(dataset_transform)
    else:
        dataset.transform = transforms.Compose(dataset_transform.append(dataset.transform))

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

    # Encoding node features into emb_dim vectors.
    # The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes=len(
        nodetypes_mapping['type']), num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)
    edge_encoder_cls = EDGE_ENCODERS[args.dataset]

    # Compute in-degree histogram over training data.
    deg = torch.zeros(800, dtype=torch.long)
    for data in dataset[split_idx['train']]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    args.deg = deg

    def run(run_id):
        best_val, final_test = 0, 0
        model = model_cls(num_tasks=len(vocab2idx), args=args, num_layer=args.num_layer, max_seq_len=args.max_seq_len, node_encoder=node_encoder, edge_encoder_cls=edge_encoder_cls, 
                        emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)

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
        best_valid_perf = eval(model, device, valid_loader, evaluator)
        best_test_perf = eval(model, device, test_loader, evaluator)
        return best_valid_perf[dataset.eval_metric], best_test_perf[dataset.eval_metric]

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
