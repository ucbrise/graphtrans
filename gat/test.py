from numpy.core.fromnumeric import mean
from tune_hyper import train

config = {"JK": 'cat', "dp_1": 0.6, "dp_2": 0.6, "gat_dp": 0.6, "gat_out_dim": 32, "lr": 0.001,
    "trans_dp": 0.3, "trans_expand": 2, "trans_head": 2, "trans_nlayer": 4, "wd": 0.0001, 'double_linear': True, 'scheduler': 'cosine'}
mean_acc = train(config)
print(mean_acc)