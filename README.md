# Representing Long-Range Context for Graph Neural Networks with Global Attention
The official code for the [paper](https://proceedings.neurips.cc//paper/2021/hash/6e67691b60ed3e4a55935261314dd534-Abstract.html)

## Installation
To setup the Python environment, please install conda first. 

All the required environments are in [requirement.yml](./requirement.yml).

```bash
conda env create -f requirement.yml
```

## How to Run

### OGBG-Code2
To run experiments:
```bash
# GraphTrans (GCN-Virtual)
python main.py --configs configs/code2/gnn-transformer/JK=cat/pooling=cls+norm_input.yml

# GraphTrans (GCN)
python main.py --configs configs/code2/gnn-transformer/no-virtual/pooling=cls+norm_input.yml

# Or to use slurm
sbatch ./slurm-run.sh configs/code2/gnn-transformer/JK=cat/pooling=cls+norm_input.yml
```

## Citation
```
@inproceedings{Wu2021GraphTrans,
  title={Representing Long-Range Context for Graph Neural Networks with Global Attention},
  author={Wu, Zhanghao and Jain, Paras and Wright, Matthew and Mirhoseini, Azalia and Gonzalez, Joseph E and Stoica, Ion},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
