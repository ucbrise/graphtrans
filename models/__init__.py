from .gnn import GNN
from .gnn_transformer import GNNTransformer
from .pna import PNANet
import functools

def partial_class(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls

def get_model_and_parser(args, parser):
    model_cls = MODELS[args.gnn_type]
    model_cls.add_args(parser)
    return model_cls

MODELS = {
    'gcn': GNN,
    'gin': GNN,
    'pna': PNANet,
    'pna-gin': partial_class(PNANet, add_edge='gin'),
    'pna-gincat': partial_class(PNANet, add_edge='gincat'),
    'gnn-transformer': partial_class(GNNTransformer),
}