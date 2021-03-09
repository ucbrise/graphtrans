import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from ogb.graphproppred.mol_encoder import AtomEncoder

import functools

NODE_ENCODERS = {
    "ogbg-code": AtomEncoder,
    "ogbg-code2": AtomEncoder,
}

EDGE_ENCODERS = {
    "ogbg-code": lambda emb_dim: nn.Linear(2, emb_dim),
    "ogbg-code2": lambda emb_dim: nn.Linear(2, emb_dim),
    "ogbg-molhiv": lambda emb_dim: BondEncoder(emb_dim=emb_dim),
}