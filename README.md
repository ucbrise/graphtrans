# Graph Augmentation
Graph augmentation/self-supervision/etc.

## Algorithms
* gcn 
* gcn+virtual node 
* gin 
* gin+virtual node
* PNA

## Augmentation methods
* None
* FLAG
* Augmentation

## Installation
To setup the Python environment, please install conda first. All the environments are in `setup.sh`.

## How to Run
To run experiments:
```bash
CUDA_VISIBLE_DEVICES=0 python src/code/main.py \
    --configs configs/code/gin-virtual/flag.yml

# Or to use slurm
sbatch ./slurm-run.sh src/code/main.py configs/code/gin-virtual/flag.yml

sbatch ./slurm-run.sh src/mol/main.py configs/molhiv/gin-virtual/flag.yml
sbatch ./slurm-run.sh src/mol/main.py configs/molpcba/gin-virtual/flag.yml
```