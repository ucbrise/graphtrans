from models.gat_transformer import GATTransformer
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models.gat import GAT
from models.gat_transformer import GATTransformer

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
print("Num Classes", dataset.num_classes)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, data = GAT(dataset).to(device), data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
model, data = GATTransformer(dataset).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-3)


def train():
    model.train()
    optimizer.zero_grad()
    F.cross_entropy(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    log_probs, accs = model(data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    result = test()
    epoch, train_acc, test_acc = result
    print(log.format(epoch, *result))
