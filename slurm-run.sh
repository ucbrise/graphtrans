#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=atlas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --exclude=freddie

# echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
# AVAILABLE_GPU=$((`gpustat | awk '{print $NF}' | grep -n "|" | cut -d : -f 1` - 2))
# CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU
script=$1
config=$2

echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
source ~/.bashrc
conda activate graph-aug

echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

echo "$script" $CUDA_VISIBLE_DEVICES
python $script --configs $config --num_workers 4 --device $CUDA_VISIBLE_DEVICES