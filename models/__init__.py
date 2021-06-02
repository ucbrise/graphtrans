from .gnn import GNN
from .gnn_transformer import GNNTransformer
from .pna_transformer import PNATransformer
from .pna import PNANet
from .transformer import Transformer


def get_model_and_parser(args, parser):
    model_cls = MODELS[args.model_type]
    model_cls.add_args(parser)
    return model_cls


MODELS = {"gnn": GNN, "pna": PNANet, "gnn-transformer": GNNTransformer, "transformer": Transformer, "pna-transformer": PNATransformer}
