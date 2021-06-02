import torch
import torch.nn as nn


class BaseModel(nn.Module):
    @staticmethod
    def need_deg():
        return False

    @staticmethod
    def add_args(parser):
        return

    @staticmethod
    def name(args):
        raise NotImplementedError

    def __init__(self):
        super().__init__()

    def forward(self, batched_data, perturb=None):
        raise NotImplementedError

    def epoch_callback(self, epoch):
        return
