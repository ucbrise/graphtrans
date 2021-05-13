#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --exclude=freddie,blaze,bombe

### SBATCH --nodelist=atlas
# echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
# AVAILABLE_GPU=$((`gpustat | awk '{print $NF}' | grep -n "|" | cut -d : -f 1` - 2))
# CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU
config=$1

echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
source ~/.bashrc
conda activate graph-aug

echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

echo "python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES"
python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES