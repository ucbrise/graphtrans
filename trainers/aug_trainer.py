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
                help="augmentation can be selected from [dnodes|pedges|subgraph|mask_nodes|none]")
        parser.add_argument('--aug_ratio', type=float, default=0.1)
        # fmt: on

    @staticmethod
    def transform(args):
        def transform_fn(data):
            aug = make_aug_list(args.aug_list)
            node_num = data.edge_index.max()

            n = np.random.randint(len(aug))
            data_aug = aug[n](data, args.aug_ratio)
            data_aug.num_nodes = data_aug.x.size(0)

            return data_aug
        return transform_fn

    @staticmethod
    def name(args):
        return "augment-" +'-'.join(sorted(args.aug_list)) + f'-{args.aug_ratio}'

def make_aug_list(aug):
    if not isinstance(aug, list):
        return [AUGMENTATIONS[aug]]
    return [AUGMENTATIONS[item] for item in aug]
