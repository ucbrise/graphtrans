# OGBG-PPA

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
python src/ppa/main.py \
    --configs configs/ppa/gin-virtual/flag.yml \
    --device 0

# Or to use slurm
sbatch ./slurm-run.sh src/ppa/main.py configs/ppa/gin-virtual/flag.yml
```