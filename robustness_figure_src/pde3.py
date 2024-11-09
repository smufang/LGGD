import torch
from torch_scatter import scatter
from torchdiffeq import odeint_adjoint as odeint


class Eikonal(torch.nn.Module):
    def __init__(self, graph, dev, mask, alpha): 
        self.graph = graph
        self.dev = dev
        self.mask = mask
        self.alpha = alpha
        self.nfe=0
        # self.p_vector = Linear(1, len(graph.x), bias=False,weight_initializer='kaiming_uniform').to(dev) 
        super(Eikonal, self).__init__()

    def grad_m_norm(self,y): #p=1
        deg = scatter(self.graph.edge_attr.view(-1,1),self.graph.edge_index[0], dim=0, dim_size=y.shape[1],reduce="add")
        # grad_m = torch.sqrt(self.graph.edge_attr.view(-1,))*(-1.0)*torch.min((y[:,self.graph.edge_index[0]]/torch.sqrt(deg.T[:,self.graph.edge_index[0]])) - (y[:, self.graph.edge_index[1]])/torch.sqrt(deg.T[:,self.graph.edge_index[1]]), torch.tensor(0.0).to(self.dev))
        # grad_m = torch.sqrt(self.graph.edge_attr.view(-1,))*(-1.0)*torch.min((y[:,self.graph.edge_index[1]]/torch.sqrt(deg.T[:,self.graph.edge_index[0]])) - (y[:, self.graph.edge_index[0]])/torch.sqrt(deg.T[:,self.graph.edge_index[0]]), torch.tensor(0.0).to(self.dev))
        # grad_m = torch.sqrt(self.graph.edge_attr.view(-1,))*(-1.0)*torch.min((y[:,self.graph.edge_index[1]]/torch.pow(deg.T[:,self.graph.edge_index[0]],1.1)) - (y[:, self.graph.edge_index[0]])/torch.pow(deg.T[:,self.graph.edge_index[0]],1.1), torch.tensor(0.0).to(self.dev))
        grad_m = torch.sqrt(self.graph.edge_attr.view(-1,))*(-1.0)*torch.min((y[:,self.graph.edge_index[1]]/torch.pow(deg.T[:,self.graph.edge_index[0]],self.alpha)) - (y[:, self.graph.edge_index[0]])/torch.pow(deg.T[:,self.graph.edge_index[0]],self.alpha), torch.tensor(0.0).to(self.dev))
        # grad_m = torch.sqrt(self.graph.edge_attr.view(-1,))*(-1.0)*torch.min((y[:,self.graph.edge_index[1]]/torch.pow(deg.T[:,self.graph.edge_index[0]],0.0)) - (y[:, self.graph.edge_index[0]])/torch.pow(deg.T[:,self.graph.edge_index[0]],0.0), torch.tensor(0.0).to(self.dev))
        # grad_m = torch.sqrt(self.graph.edge_attr.view(-1,))*torch.max(y[:,self.graph.edge_index[1]] - y[:, self.graph.edge_index[0]], torch.tensor(0.0).to(self.dev))
        # print(f"grad_m.shape = {grad_m.shape}")
        # grad_norm = scatter(torch.abs(grad_m),self.graph.edge_index[0],dim=1, dim_size=y.shape[1],reduce="add")
        grad_norm = scatter(torch.abs(grad_m),self.graph.edge_index[0],dim=1, dim_size=y.shape[1],reduce="max")
        # print(f"self.graph_m_norm.shape = {grad_norm.shape}")
        return grad_norm

    def forward(self, t, y):
        self.nfe += 1
        deg = scatter(self.graph.edge_attr.view(-1,1),self.graph.edge_index[0], dim=0, dim_size=y.shape[1],reduce="add")
        deg = deg.view(1,-1)
        # deg = scatter(self.graph.edge_attr.view(-1,1),self.graph.edge_index[0], dim=0, dim_size=y.shape[1],reduce="add")
        #print(deg.shape)
        #print(y.shape)
        # Eikonal = (scatter(self.graph.edge_attr.view(-1,)*((y[:,self.graph.edge_index[1]]/torch.sqrt(deg.T[:,self.graph.edge_index[1]] * deg.T[:,self.graph.edge_index[0]]))-(y[:,self.graph.edge_index[0]]/deg.T[:,self.graph.edge_index[0]])), self.graph.edge_index[0],dim=1, dim_size=y.shape[1],reduce="add"))
        # Eikonal = (scatter(self.graph.edge_attr.view(-1,)*(y[:,self.graph.edge_index[1]]-y[:,self.graph.edge_index[0]]), self.graph.edge_index[0],dim=1, dim_size=y.shape[1],reduce="add"))
        f = 1.0 - self.grad_m_norm(y)
        # f = torch.pow(deg, 1.0) - self.grad_m_norm(y)
        # return f
        # if not self.run_pde:
        #     return  f
        # else:
        return f * self.mask
        # return f

class EikonalBlock(torch.nn.Module):
    def __init__(self, odefunc,t, dev, **kwargs):
        super(EikonalBlock, self).__init__()
        self.odefunc = odefunc
        self.t = t
        self.rtol = kwargs["rtol"]
        self.dev = dev

    def nfe(self):
        return self.odefunc.nfe

    def forward(self, x):
        # z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10).to(self.dev)
        # z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10, method="rk4", options={"step_size":0.01}).to(self.dev)
        z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10, method="rk4", options={"step_size":0.1}).to(self.dev)
        # z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10, method="rk4", options={"step_size":1.0}).to(self.dev)
        return  z[1]

class Net(torch.nn.Module):
    def __init__(self, graph, mask, front_initial, alpha, time, dev):
        super(Net, self).__init__()
        self.mask = mask.T
        self.graph = graph
        self.front_initial = front_initial
        self.dev = dev
        self.alpha = alpha 
        # self.eikonalblock = EikonalBlock(Eikonal(self.graph), t=time, **dict(rtol=0.00264))
        # self.eikonalblock = EikonalBlock(Eikonal(self.graph), t=time, **dict(rtol=1))
        self.eikonalblock = EikonalBlock(Eikonal(self.graph,self.dev,self.mask, alpha=self.alpha), t=time, dev=self.dev, **dict(rtol=0.001))
    
    def nfe(self):
        return self.eikonalblock.nfe()

    def forward(self,x):
        #self.fzero = (x.T * self.mask + self.front_initial).T
        #self.fzero = (self.front_initial).T
        # self.fzero = self.front_initial
        # z = self.eikonalblock(self.fzero.T)
        z = self.eikonalblock(self.front_initial.T)
        # z = self.eikonalblock(x.T)
        return z
