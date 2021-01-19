import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from trainers import register_trainer
import numpy as np
from copy import deepcopy
from data.augmentation import AUGMENTATIONS

@register_trainer("augment")
class AugTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--aug_list', type=str, nargs='+', default=["none"],
                help="augmentation can be selected from [nnodes|pedges|subgraph|mask_nodes|none]")
        # fmt: on

    @staticmethod
    def transform(args):
        def transform_fn(data):
            aug = make_aug_list(args.aug_list)
            node_num = data.edge_index.max()
            # sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            # data.edge_index = torch.cat((data.edge_index, sl), dim=1)

            n = np.random.randint(len(aug))
            data_aug = aug[n](deepcopy(data))

            # edge_idx = data_aug.edge_index.numpy()
            # _, edge_num = edge_idx.shape
            # idx_not_missing = sorted(list(set(edge_idx[0]) | set(edge_idx[1])))
            
            # node_num_aug = len(idx_not_missing)
            # data_aug.x = data_aug.x[idx_not_missing]

            # # data_aug.batch = data.batch[idx_not_missing]
            # idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}

            # edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num)]
            # data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            # # print(data, data_aug)
            # # assert False

            return data_aug
        return transform_fn

def make_aug_list(aug):
    if not isinstance(aug, list):
        return [AUGMENTATIONS[aug]]
    return [AUGMENTATIONS[item] for item in aug]
