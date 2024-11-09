import os 
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import is_undirected, to_undirected
from utils import *

class Graph():
    def __init__(self, args):
        self.args = args
        self.dataset = Planetoid(root="data", name="Cora")
        self.graph = self.dataset[0]

    def gen_split(self):
        self.graph.train_mask, self.graph.val_mask, self.graph.t3_mask, self.graph.t4_mask, self.graph.t5_mask,  self.graph.test_mask = gen_mask_multi(self.graph.y, self.args.train_p, self.args.val_p, 10, 10, 10, self.dataset.num_classes)

    def get_graph(self):
        self.gen_split()
        self.graph.edge_index = to_undirected(self.graph.edge_index)
        self.graph.edge_attr = torch.ones(self.graph.edge_index.shape[1],1)
        return self.graph
