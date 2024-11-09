import os 
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from utils import *

class Graph():
    def __init__(self, args):
        self.args = args
        self.dataset = Planetoid(root="data", name="Cora")
        self.graph = self.dataset[0]
        self.graph.num_classes = self.dataset.num_classes

    def gen_split(self):
        self.graph.seed_mask, self.graph.train_mask, self.graph.val_mask, self.graph.test_mask = gen_mask_p(self.graph.y, 0, self.args.train_p, self.args.val_p, self.dataset.num_classes)

    def get_graph(self):
        transform = ToUndirected()
        self.graph = transform(self.graph)
        self.gen_split()
        self.graph.edge_attr = torch.ones(self.graph.edge_index.shape[1],1)
        return self.graph
