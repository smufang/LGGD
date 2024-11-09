import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from utils import *
import random
import numpy as np
import argparse
import random
import argparse
import torch
import numpy as np
import torch_geometric
from tqdm import tqdm
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import is_undirected, to_undirected
from torchdiffeq import odeint_adjoint as odeint
from torch.nn import ReLU, Sigmoid, Dropout, Tanh
from torch_geometric.nn.dense.linear import Linear
from new_utils import get_front_eikonal
from model_val import GCN, Net
from data_loader_val import Graph
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rs", type=int, default=0, help="random_seed")
    parser.add_argument("--hs", type=int, default=32, help="hidden layer size")
    parser.add_argument("--wd", type=float, default=0.005, help="weight decay L2")
    parser.add_argument("--wd2", type=float, default=1e-6, help="weight decay L2")
    parser.add_argument("--alpha1", type=float, default=0.5, help="alpha in dropout")
    parser.add_argument("--alpha2", type=float, default=0.5, help="alpha in dropout")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha in pde")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01, help="rtol in diffeq")
    parser.add_argument("--itr", type=int, default=100)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--nowgts", action='store_true')
    parser.add_argument("--dev", type=str, default="cuda:0")
    parser.add_argument("--total_mask", type=int, default=0)
    parser.add_argument("--train_p", type=float, default=2.5)
    parser.add_argument("--val_p", type=float, default=2.5)
    parser.add_argument("--saved_front", type=int, default=0)
    parser.add_argument("--hs1", type=float, default=512)
    parser.add_argument("--hs2", type=float, default=128)
    args = parser.parse_args()

    random.seed(args.rs)
    np.random.seed(args.rs)
    torch.manual_seed(args.rs)
    torch.cuda.manual_seed(args.rs)
    torch.cuda.manual_seed_all(args.rs)

    graph = Graph(args)
    graph = graph.get_graph()
    graph = graph.to(args.dev)

    num_dist = int(args.time)
    for i in range(num_dist):
        args.time = float(i+1) 
        model = Net(args, args.alpha, graph, args.dev, **dict(nowgts=args.nowgts, lamb=args.lamb, inp=graph.x.shape[1],out=(graph.y.max()+1).item(),rtol=args.rtol, alpha1=args.alpha1, alpha2=args.alpha2)).to(args.dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model.configure(opt)
        model.fit()
        model.save_dist()
        model.save_front()

    graph = model.graph_out2()
    model = GCN(graph, graph.x.shape[1], args.hs, (graph.y.max()+1).item()).to(args.dev)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=args.wd2)
    model.configure(optimizer)
    model.fit()
    print(f"gcn wgts are saved")

    args.saved_front = 1
    graph.sum_mask = torch.zeros(len(graph.y), dtype=torch.bool).to(args.dev)
    accs = []
    for _, mask in graph('train_mask', 't3_mask', 't4_mask', 't5_mask'):
        graph.sum_mask = graph.sum_mask + mask
        for i in range(num_dist):
            args.time = float(i+1) 
            model = Net(args, args.alpha, graph, args.dev, **dict(nowgts=args.nowgts, lamb=args.lamb, inp=graph.x.shape[1],out=(graph.y.max()+1).item(),rtol=args.rtol, alpha1=args.alpha1, alpha2=args.alpha2)).to(args.dev)
            model.save_dist2() # withoug calling the fit function.
        print(f"generated features for boundary size={math.ceil(torch.sum(graph.sum_mask)*100/len(graph.sum_mask))} %")

        graph = model.graph_out2() # concatenate the distances
        model = torch.load("downstream.pth")
        with torch.no_grad():
            model.eval()
            logits = model.forward_multi(graph)
        pred = logits[graph.test_mask].max(1)[1]
        acc = pred.eq(graph.y[graph.test_mask]).sum().item() / graph.test_mask.sum().item()
        accs.append(acc)
    print(f"done everything, here are the accuracies:")
    print(accs)