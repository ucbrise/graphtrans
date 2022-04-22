# Representing Long-Range Context for Graph Neural Networks with Global Attention
```
@inproceedings{Wu2021GraphTrans,
  title={Representing Long-Range Context for Graph Neural Networks with Global Attention},
  author={Wu, Zhanghao and Jain, Paras and Wright, Matthew and Mirhoseini, Azalia and Gonzalez, Joseph E and Stoica, Ion},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
## Overview
We release the PyTorch code for the GraphTrans [[paper](https://proceedings.neurips.cc//paper/2021/hash/6e67691b60ed3e4a55935261314dd534-Abstract.html)]

## Installation
To setup the Python environment, please install conda first. 
All the required environments are in [requirement.yml](./requirement.yml).
```bash
conda env create -f requirement.yml
```
## How to Run

To run the experiments, please refer to the commands below (taking OGBG-Code2 as an example):
```bash
# GraphTrans (GCN-Virtual)
python main.py --configs configs/code2/gnn-transformer/JK=cat/pooling=cls+norm_input.yml --runs 5
# GraphTrans (GCN)
python main.py --configs configs/code2/gnn-transformer/no-virtual/pooling=cls+norm_input.yml --runs 5
# Or to use slurm
sbatch ./slurm-run.sh ”configs/code2/gnn-transformer/JK=cat/pooling=cls+norm_input.yml --runs 5”
```
The config path for each dataset/model can be found in the result table below.
## Results
| Dataset | Model | Valid | Test | Config |
|:--|:--|:--:|:--:|:--:|
| [OGBG-Code2](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-code2) | GraphTrans (GCN) | 0.1599±0.0009 | 0.1751±0.0015 | [Config](configs/code2/gnn-transformer/no-virtual/pooling=cls+norm_input.yml) |
| | GraphTrans (PNA) | 0.1622±0.0025 | 0.1765±0.0033 | [Config](configs/code2/pna-transformer/pooling=cls+norm_input.yml) |
| | GraphTrans (GCN-Virtual) | 0.1661±0.0012 | 0.1830±0.0024 | [Config](configs/code2/gnn-transformer/JK=cat/pooling=cls+norm_input.yml) |
| [OGBG-Molpcba](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molpcba) | GraphTrans (GIN) | 0.2893±0.0050 | 0.2756±0.0039 | [Config](configs/molpcba/gnn-transformer/no-virtual/JK=cat/pooling=cls+gin+norm_input.yml) |
| | GraphTrans (GIN-Virtual) | 0.2867±0.0022 | 0.2761±0.0029 | [Config](configs/molpcba/gnn-transformer/JK=cat/pooling=cls+gin+norm_input.yml) |
| [NCI1](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) | GraphTrans (small, GCN) | — | 81.3±1.9 | [Config](configs/NCI1/gnn-transformer/no-virtual/gd=128+gdp=0.1+tdp=0.1+l=3+cosine.yml) |
| | GraphTrans (large, GIN) | — | 82.6±1.2 | [Config](configs/NCI1/gnn-transformer/no-virtual/gin+gdp=0.1+tdp=0.1+l=4+cosine.yml) |
| [NCI109](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) | GraphTrans (small, GCN) | — | 79.2±2.2 | [Config](configs/NCI109/gnn-transformer/no-virtual/ablation-pos_encoder) |
| | GraphTrans (large, GIN) | — | 82.3±2.6 | [Config](configs/NCI109/gnn-transformer/no-virtual/gin+gdp=0.1+tdp=0.1+l=4+cosine.yml) |


