# OGBG-Code

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
python srccode/main.py \
    --configs configs/code/gin-virtual/flag.yml \
    --device 0

# Or to use slurm
sbatch ./slurm-run.sh src/code/main.py configs/code/gin-virtual/flag.yml
```