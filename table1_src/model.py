import random
import argparse
import torch
import numpy as np
import torch_geometric
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import is_undirected, to_undirected
from torchdiffeq import odeint_adjoint as odeint
from torch.nn import ReLU, Sigmoid, Dropout, Tanh, ELU
from torch_geometric.nn.dense.linear import Linear
from new_utils import *
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm


class Eikonal(torch.nn.Module):
    def __init__(self, alpha, graph, dev, mask, lamb, nowgts):
        super(Eikonal, self).__init__()
        self.run_pde = None
        self.alpha = alpha
        self.mask = mask
        self.graph = graph
        self.dev = dev
        self.distances = Linear(1, self.graph.edge_index.shape[1], bias=False,weight_initializer='kaiming_uniform').to(self.dev) 
        self.rel = Sigmoid()
        self.lamb = lamb
        self.nowgts = nowgts
        self.p_vec = Linear(1, len(graph.x), bias=False, weight_initializer='kaiming_uniform').to(self.dev) 

    def grad_m_norm(self,y): #p=1
        p_vec = self.rel(self.p_vec.weight)
        edge_attr = self.rel(self.distances.weight)
        deg = scatter(self.graph.edge_attr.view(-1,1),self.graph.edge_index[0], dim=0, dim_size=y.shape[1],reduce="add")
        grad_m = torch.sqrt(edge_attr.view(-1,))*(-1.0)*torch.min((y[:,self.graph.edge_index[1]]/torch.pow(deg.T[:,self.graph.edge_index[0]],self.alpha)) - (y[:, self.graph.edge_index[0]])/torch.pow(deg.T[:,self.graph.edge_index[0]],self.alpha), torch.tensor(0.0).to(self.dev))
        grad_norm = scatter(torch.abs(grad_m),self.graph.edge_index[0],dim=1, dim_size=y.shape[1],reduce="add")
        return grad_norm

    def forward(self, t, y):
        f = 1.0 - self.grad_m_norm(y)
        if not self.run_pde:
            return  f
        else:
            return f * self.mask

class EikonalBlock(torch.nn.Module):
    def __init__(self, odefunc,t, dev, **kwargs):
        super(EikonalBlock, self).__init__()
        self.odefunc = odefunc
        self.t = t
        self.rtol = kwargs["rtol"]
        self.dev = dev

    def run_pde(self):
        self.odefunc.run_pde = True

    def donot_run_pde(self):
        self.odefunc.run_pde = False

    def forward(self, x):
        z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10, method="rk4", options={"step_size":0.1}).to(self.dev)
        # z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10, method="rk4", options={"step_size":1.0}).to(self.dev)
        return  z[1]

class Net(torch.nn.Module):
    # def __init__(self, args, alpha, graph, mask, front_initial, time, dev, **kwargs):
    def __init__(self, args, alpha, graph,  dev, **kwargs):
        super(Net, self).__init__()
        self.run_pde = False
        self.fzero = None
        self.alpha = alpha
        self.graph = graph
        self.args = args
        self.dev = dev
        self.s_vec = Linear(1,kwargs["out"],bias=False,weight_initializer='kaiming_uniform').to(self.dev)
        self.m1 = Linear(kwargs["inp"],512,weight_initializer='kaiming_uniform')
        self.m2 = Linear(512,128,weight_initializer='kaiming_uniform')
        self.m3 = Linear(128,kwargs["out"], weight_initializer='kaiming_uniform')
        self.dropout1 = torch.nn.Dropout(kwargs["alpha1"])
        self.dropout2 = torch.nn.Dropout(kwargs["alpha2"])
        self.rel = ReLU()
        # self.rel = ELU()
        # self.front_initial = front_initial
        self.front_initial = get_front_eikonal(self.graph.y, self.graph.train_mask)
        self.mask = torch.where(self.front_initial==0, False, True)
        self.time = (torch.linspace(0,args.time,2)).to(args.dev)
        self.sig = Sigmoid()
        self.eikonalblock = EikonalBlock(Eikonal(self.alpha, self.graph, self.dev, self.mask, kwargs["lamb"], kwargs["nowgts"]), t=self.time, dev=self.dev, **dict(rtol=kwargs["rtol"]))

    def forward(self,x):
        x = self.m1(x)
        x = self.rel(x)
        x = self.dropout1(x)
        x = self.m2(x)
        x = self.rel(x)
        x = self.dropout2(x)
        x = self.m3(x)
        # x = self.sig(x) #BIG DIFF comp to RELU
        # x = self.rel(x)
        x = torch.abs(x)
        if not self.run_pde:
            self.eikonalblock.donot_run_pde()
            z = self.eikonalblock(x.T)
            # z = (z.T * s_vec.T).T
            return -x, -z
        else:
            # self.fzero = (x.T * self.mask + self.front_initial).T
            self.fzero = (x.T * self.mask).T
            self.eikonalblock.run_pde()
            z = self.eikonalblock(self.fzero.T)
            # z = (z.T * s_vec.T).T
            # z = self.eikonalblock(self.front_initial)
            # z = self.eikonalblock(x.T)
            return -x, -z

    def configure(self, opt):
        self.opt = opt

    def trainer(self):
        self.train()
        self.run_pde = False
        out_front, out = self(self.graph.x)
        loss =  torch.nn.CrossEntropyLoss()(out.T[self.graph.train_mask], self.graph.y[self.graph.train_mask]) 
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def test(self):
        self.eval()
        self.run_pde = True
        out_front, out = self(self.graph.x)

        accs = []
        a = torch.argmax(out.T, dim=1)
        nmask_train = (a[self.graph.train_mask] == self.graph.y[self.graph.train_mask] )
        train_acc = (torch.sum(nmask_train))/len(self.graph.y[self.graph.train_mask])
        accs.append(train_acc)

        nmask_val = (a[self.graph.val_mask] == self.graph.y[self.graph.val_mask] )
        val_acc = (torch.sum(nmask_val))/len(self.graph.y[self.graph.val_mask])
        accs.append(val_acc)

        nmask_test = (a[self.graph.test_mask] == self.graph.y[self.graph.test_mask] )
        test_acc = (torch.sum(nmask_test))/len(self.graph.y[self.graph.test_mask])
        accs.append(test_acc)

        loss =  torch.nn.CrossEntropyLoss()(out.T[self.graph.val_mask], self.graph.y[self.graph.val_mask]) 
        accs.append(loss.item())
        accs.append(out_front)
        return accs

    def fit(self):
        early_stopping_counter = 100
        best_val_acc = test_acc = 0
        best_val_loss = np.inf
        for itr in tqdm(range(1, self.args.itr)):
            self.trainer()
            train_acc, val_acc, tmp_test_acc, val_loss, out_front = self.test()
            if (val_loss < best_val_loss):
            # if (val_acc > best_val_acc):
                best_val_acc = val_acc
                best_val_loss = val_loss
                test_acc = tmp_test_acc
                early_stopping_counter = 100
                torch.save(self, 'eiko-pde5.pth')
            else:
                early_stopping_counter -= 1
                if early_stopping_counter == 0:
                    break

    def graph_out2(self):
        test1 = torch.load("test1.pt")
        test2 = torch.load("test2.pt")
        test3 = torch.load("test3.pt")
        test4 = torch.load("test4.pt")
        test5 = torch.load("test5.pt")
        testf = torch.cat([test1,test2,test3,test4,test5],dim=1)
        testf = testf.to(self.args.dev)
        self.graph.x = testf
        return self.graph

    def save_dist(self):
        new_model = torch.load("eiko-pde5.pth")
        with torch.no_grad():
            new_model.eval()
            new_model.run_pde = True
            out_front, out = new_model(self.graph.x)
        test = torch.min(out.T, dim=0)[0]
        test2 = out.T/(test.view(1,-1)) 
        torch.save((-1.0)*test2.cpu(), f"test{int(self.args.time)}.pt")
        print(f"learned eikonal distance  t={int(self.args.time)}")
        return


class GCN(torch.nn.Module):
    def __init__(self, graph, in_features, hidden_features, num_classes):
        super(GCN, self).__init__()
        self.graph = graph
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    def forward(self):
        x, edge_index, edge_attr = self.graph.x, self.graph.edge_index, self.graph.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def configure(self, opt):
        self.opt = opt

    def trainer(self): 
        self.train()
        self.opt.zero_grad()
        out = self()
        loss = F.nll_loss(out[self.graph.train_mask], self.graph.y[self.graph.train_mask])
        loss.backward()
        self.opt.step()

    def evaluate(self):
        self.eval()
        with torch.no_grad():
            logits, accs, losses = self(), [], []
        for _, mask in self.graph('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(self.graph.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(self()[mask], self.graph.y[mask])
            accs.append(acc)
            losses.append(loss)
        return accs, losses

    def fit(self):
        best_val_acc = test_acc = best_test_acc = 0
        patience_counter = 100
        best_val_loss = float('inf')
        for epoch in range(1, 5000):
            self.trainer()
            [train_acc, val_acc, test_acc], [train_loss, val_loss, test_loss] = self.evaluate()
            if val_acc > best_val_acc:
            # if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_test_acc = test_acc
                patience_counter = 100
                #torch.save will come here
            else:
                patience_counter -= 1
                if patience_counter == 0:
                    break
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best_Val: {best_val_acc:.4f}, Best_Test: {best_test_acc:.4f}, Val_loss: {val_loss:.4f}')
