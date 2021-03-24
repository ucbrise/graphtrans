from .gnn import GNN
from .gnn_transformer import GNNTransformer
from .pna import PNANet
from .multibranch import MultiBranch
import functools

def get_model_and_parser(args, parser):
    model_cls = MODELS[args.model_type]
    model_cls.add_args(parser)
    return model_cls

MODELS = {
    'gnn': GNN,
    'pna': PNANet,
    'gnn-transformer': GNNTransformer,
    'multibranch': MultiBranch
}