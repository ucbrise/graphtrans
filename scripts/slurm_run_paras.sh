#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --exclude=freddie,blaze,bombe

config=$1

eval "$(conda shell.bash hook)"
echo $(scontrol show hostnames $SLURM_JOB_NODELIST)

source activate graph-aug

set -x
set -e
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

echo "python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES"
python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES