import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from typing import Any
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split
from model import Encoder, Model, drop_feature, SAGEncoder
from eval import label_classification, MulticlassEvaluator, log_regression, print_gpu_used_info
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code', 'ogbn-products']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        # return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name)

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def generate_split(num_samples: int, train_ratio: float, val_ratio: float, ogb_split_idx: Any):
    if ogb_split_idx is not None:
        idx_train = ogb_split_idx['train']
        idx_val = ogb_split_idx['valid']
        idx_test = ogb_split_idx['test']
    else:
        train_len = int(num_samples * train_ratio)
        val_len = int(num_samples * val_ratio)
        test_len = num_samples - train_len - val_len
        train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))
        idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
        
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_ogb(model: Model, x, edge_index, split_idx):
    model.train()
    optimizer.zero_grad()
    train_idx = split_idx['train']
    train_loader = NeighborSampler(edge_index.to(device), node_idx=train_idx.to(device),
                                sizes=[15, 10, 5], batch_size=1024,
                                shuffle=True, num_workers=12)
    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs1 = []
        adjs2 = []
        adjs = [adj.to(device) for adj in adjs]
        for adj in adjs:
            e_index, _, size = adj
            edge_index_1 = dropout_adj(e_index, p=drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(e_index, p=drop_edge_rate_2)[0]
            adj_1 = EdgeIndex(edge_index_1, _, size)
            adj_2 = EdgeIndex(edge_index_2, _, size)
            adjs1.append(adj_1)
            adjs2.append(adj_2)           
        
        x_1 = drop_feature(x[n_id], drop_feature_rate_1)
        x_2 = drop_feature(x[n_id], drop_feature_rate_2)
        z1 = model(x_1.to(device), adjs1)
        z2 = model(x_2.to(device), adjs2)
        loss = model.loss(z1, z2, batch_size=0)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        pbar.update(batch_size)
    pbar.close()
    loss = total_loss / len(train_loader)
    return loss
    # model.train()
    # optimizer.zero_grad()
    
    # edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    # edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    # x_1 = drop_feature(x, drop_feature_rate_1)
    # x_2 = drop_feature(x, drop_feature_rate_2)

    # train_idx = split_idx['train']
    # total_loss = 0
    # train_loader1 = NeighborSampler(edge_index_1.to(device), node_idx=train_idx.to(device),
    #                             sizes=[15, 10, 5], batch_size=1024,
    #                             shuffle=False, num_workers=12)
    # train_loader2 = NeighborSampler(edge_index_2.to(device), node_idx=train_idx.to(device),
    #                             sizes=[15, 10, 5], batch_size=1024,
    #                             shuffle=False, num_workers=12)

    # for data1, data2 in zip(train_loader1, train_loader2):
    #     batch_size1, n_id1, adjs1 = data1
    #     adjs1 = [adj.to(device) for adj in adjs1]
    #     batch_size2, n_id2, adjs2 = data2
    #     adjs2 = [adj.to(device) for adj in adjs2]
    #     z1 = model(x_1[n_id1].to(device), adjs1)
    #     z2 = model(x_2[n_id2].to(device), adjs2)
    #     loss = model.loss(z1, z2, batch_size=0)
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += float(loss)
    # loss = total_loss / len(train_loader1)
    # return loss


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)
    

def test_ogb(model, data, evaluator, split):
    model.eval()
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                batch_size=64, shuffle=False,
                                num_workers=0)
    print('######################Before inference######################')
    print_gpu_used_info()
    out = model.inference(data.x, subgraph_loader, device)
    acc = log_regression(out, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=split)['acc']
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)
    
    split_idx = None
    if args.dataset in ('ogbn-arxiv', 'ogbn-products'):
        split_idx = dataset.get_idx_split()
    
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1, ogb_split_idx=split_idx)
    
    encoder = SAGEncoder(dataset.num_features, num_hidden, num_proj_hidden, 3).to(device)

    # encoder = Encoder(dataset.num_features, num_hidden, activation,
    #                   base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    evaluator = None
    if args.dataset in ('ogbn-arxiv', 'ogbn-products'):
        # evaluator = Evaluator(name='ogbn-arxiv')
        evaluator = MulticlassEvaluator()

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        # loss = train(model, data.x, data.edge_index)
        loss = train_ogb(model, data.x, data.edge_index, split_idx)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        
        if epoch % 1 == 0:
            if args.dataset in ('ogbn-arxiv', 'ogbn-products'):
                acc = test_ogb(model, data, evaluator, split)
                print(f'Epoch: {epoch:02d}, ',
                      f'Acc: {100 * acc:.2f}%')

    print("=== Final ===")
    if args.dataset in ('ogbn-arxiv', 'ogbn-products'):
        acc = test_ogb(model, data, evaluator, split)
        print(f'Acc: {100 * acc:.2f}%')
    else:
        test(model, data.x, data.edge_index, data.y, final=True)
