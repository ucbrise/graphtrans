# OGBG-MOL[HIV|PCBA]

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
python src/mol/main.py \
    --configs configs/molhiv/gin-virtual/flag.yml \
    --device 0

python src/mol/main.py \
    --configs configs/molpcba/gin-virtual/flag.yml \
    --device 0

# Or to use slurm
sbatch ./slurm-run.sh src/mol/main.py configs/molhiv/gin-virtual/flag.yml
sbatch ./slurm-run.sh src/mol/main.py configs/molpcba/gin-virtual/flag.yml
```