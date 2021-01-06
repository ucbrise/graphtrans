#!/bin/bash
source ~/.bashrc
conda create -n graph-aug python=3.8
conda activate graph-aug
conda install pytorch=1.7 torchvision torchaudio cudatoolkit=10.2 -c pytorch -y

TORCH=1.7.0
CUDA=cu102
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install ogb

conda install --name graph-aug pylint -y
conda install --name graph-aug yapf -y