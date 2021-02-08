from .gnn import GNN
from .pna import PNANet
import functools

def partial_class(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls

def get_model_and_parser(args, parser):
    gnn_split = args.gnn.split('-')
    if gnn_split[0] in ['gin', 'gcn']:
        if len(gnn_split) > 1:
            assert gnn_split[1] == 'virtual'
            return partial_class(GNN, gnn_type=gnn_split[0], virtual_node=True)
        else:
            return partial_class(GNN, gnn_type=gnn_split[0])
    elif gnn_split[0] == 'pna':
        PNANet.add_args(parser)
        if len(gnn_split) > 1:
            return partial_class(PNANet, add_edge=gnn_split[1])
        return PNANet
    else:
        raise ValueError('Invalid GNN type')
