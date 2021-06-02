# Graph Augmentation
Graph augmentation/self-supervision/etc.

## Algorithms
* gcn 
* gcn+virtual node 
* gin 
* gin+virtual node
* PNA
* GraphTrans

## Augmentation methods
* None
* FLAG
* Augmentation

## Installation
To setup the Python environment, please install conda first. 

All the required environments are in [setup.sh](./setup.sh).

## How to Run
To run experiments:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --configs configs/code2/gcn-virtual/baseline+run1+seed.yml

# Or to use slurm
sbatch ./slurm-run.sh configs/code2/gcn-virtual/baseline+run1+seed.yml
```

## Exps
### GNN-Transformer
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --configs configs/code2/gnn-transformer/JK=cat/pooling=cls+norm_input.yml
```