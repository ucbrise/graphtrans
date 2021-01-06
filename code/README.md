# OGBG-Code

## Algorithms
* gcn 
* gcn+virtual node 
* gin 
* gin+virtual node

## Augmentation methods
* None
* FLAG

## Installation
To setup the Python environment, please install conda first. All the environments are in `setup.sh`.

## How to Run
To run experiments:
```bash
python code/main.py \
    --configs configs/code/gin-virtual/flag.yml \
    --device 0

# Or to use slurm
sbatch ./slurm-run.sh code/main.py configs/code/gin-virtual/flag.yml
```