import torch
import torchvision
from collections import Counter
import torchvision.transforms as transforms

def getWgStats(wg):
    return f"The max is {torch.max(wg)}; The min is {torch.min(wg)}; The std is {torch.std(wg)};  The median is {torch.median(wg)}; The mean is {torch.mean(wg)}"

def gen_mask_p(labels, seed_size_p=0, train_size_p=10, val_size_p=10, classes=10):
    """
    Hmm, I can do better than this.
    """
    new_seed_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_train_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_val_mask = torch.zeros(len(labels), dtype=torch.bool) #80
    count = Counter(labels.tolist())
    for i in range(0,int(classes)):
        full_mask = (labels == i)
        ids = torch.where(full_mask == True)[0]
        #train_mask = ids[torch.randperm(len(ids))[:20]]

        seed_size = int((seed_size_p * count[i])//100)
        train_size = int((train_size_p * count[i])//100)
        val_size = int((val_size_p * count[i])//100)

        #print("hello", train_size+val_size+seed_size)
        indices = ids[torch.randperm(len(ids))[:train_size+val_size+seed_size]]
        new_seed_mask[indices[:seed_size]] = True
        new_train_mask[indices[seed_size:seed_size+train_size]] = True
        new_val_mask[indices[seed_size+train_size:seed_size+train_size+val_size]] = True
    new_test_mask = ~(new_seed_mask + new_train_mask + new_val_mask)
    assert (torch.sum(new_test_mask) + torch.sum(new_train_mask) + 
    torch.sum(new_val_mask) + torch.sum(new_seed_mask) == len(labels))
    return new_seed_mask, new_train_mask, new_val_mask, new_test_mask 

def gen_mask_multi(labels, t1_size_p=0, t2_size_p=10, t3_size_p=10, t4_size_p=10, t5_size_p=10, classes=10):
    """
    Hmm, I can do better than this.
    """
    new_t1_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_t2_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_t3_mask = torch.zeros(len(labels), dtype=torch.bool) #80
    new_t4_mask = torch.zeros(len(labels), dtype=torch.bool) #80
    new_t5_mask = torch.zeros(len(labels), dtype=torch.bool) #80
    count = Counter(labels.tolist())
    for i in range(0,int(classes)):
        full_mask = (labels == i)
        ids = torch.where(full_mask == True)[0]
        #train_mask = ids[torch.randperm(len(ids))[:20]]

        t1_size = int((t1_size_p * count[i])//100)
        t2_size = int((t2_size_p * count[i])//100)
        t3_size = int((t3_size_p * count[i])//100)
        t4_size = int((t4_size_p * count[i])//100)
        t5_size = int((t5_size_p * count[i])//100)

        #print("hello", train_size+val_size+seed_size)
        # indices = ids[torch.randperm(len(ids))[:train_size+val_size+seed_size]]
        indices = ids[torch.randperm(len(ids))[:t1_size+t2_size+t3_size+t4_size+t5_size]]
        new_t1_mask[indices[:t1_size]] = True
        new_t2_mask[indices[t1_size:t1_size+t2_size]] = True
        new_t3_mask[indices[t1_size+t2_size:t1_size+t2_size+t3_size]] = True
        new_t4_mask[indices[t1_size+t2_size+t3_size:t1_size+t2_size+t3_size+t4_size]] = True
        new_t5_mask[indices[t1_size+t2_size+t3_size+t4_size:t1_size+t2_size+t3_size+t4_size+t5_size]] = True
    # new_test_mask = ~(new_seed_mask + new_train_mask + new_val_mask)
    new_test_mask = ~(new_t1_mask + new_t2_mask + new_t3_mask + new_t4_mask + new_t5_mask)
    assert (torch.sum(new_test_mask) + torch.sum(new_t1_mask) + torch.sum(new_t2_mask) + torch.sum(new_t3_mask) + torch.sum(new_t4_mask) + torch.sum(new_t5_mask) == len(labels))
    return new_t1_mask, new_t2_mask, new_t3_mask, new_t4_mask, new_t5_mask, new_test_mask

def get_sim_euc(data,sig):
    data.edge_attr = torch.exp( ( (-1.0) * torch.norm(data.x[data.edge_index[0]] - data.x[data.edge_index[1]], p=2, dim=1)**2)/sig)
    print(f"shape of dataedge attr: {data.edge_attr.shape}")
    print(data.x[data.edge_index[0]].shape)
    return


def pearson_correlation(x, y):
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    dot_product = torch.sum(x_centered * y_centered, dim=1)
    x_magnitude = torch.norm(x_centered, dim=1)
    y_magnitude = torch.norm(y_centered, dim=1)
    correlation = dot_product / (x_magnitude * y_magnitude)
    return correlation


def get_sim_corr(data, sig):
    correlation = pearson_correlation(data.x[data.edge_index[0]],data.x[data.edge_index[1]]) 
    data.edge_attr = torch.exp(((-1.0)*(1.0 - correlation)**2)/(sig))
    return


def get_sim_corr2(data, eps):
    correlation = pearson_correlation(data.x[data.edge_index[0]],data.x[data.edge_index[1]]) 
    # data.edge_attr = torch.exp(((-1.0)*(1.0 - correlation)**2)/(sig))
    data.edge_attr = 1.0/(1.0 - correlation + eps)
    return

from torch_geometric.utils import to_scipy_sparse_matrix
def get_A_calder(data):
    A_calder = to_scipy_sparse_matrix(data.edge_index, data.edge_attr)
    return A_calder
    

from torch_geometric.utils import to_dense_adj
def get_A_calder_2(data):
    A_calder = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)
    return A_calder.cpu().numpy()
    



from torch_geometric.utils import add_self_loops
def add_loops(data):
    data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr, num_nodes=len(data.x))
    return


def zero_check(data, dev):
    data.edge_attr = torch.where(torch.isnan(data.edge_attr), torch.tensor(0.0).to(dev), data.edge_attr)
    pass


from torch_scatter import scatter
def deg_normalize(data):
    deg = scatter(data.edge_attr.view(-1,1),data.edge_index[0],dim=0, dim_size=data.x.shape[0], reduce="add")
    tmp = deg[data.edge_index[0]]
    data.edge_attr = data.edge_attr/tmp.view(-1, )
    return

def wgts_k_thresh(data, k_t, dev):
    copy = data.edge_attr.clone().detach()
    print("hello, I am heereeeeeee")
    print(data.edge_attr.shape)
    print(data.edge_index.shape)
    print(copy.shape)
    for i in range(k_t):
        print(f"ehllo the i value is {i}")
        max_k = scatter(copy.view(-1,1),data.edge_index[0],dim=0, dim_size=data.x.shape[0], reduce="max")
        print(f"hello max_k shape is {max_k.shape}")
        copy = torch.where(copy.view(-1,) == max_k[data.edge_index[0]].view(-1,), torch.tensor(0.0).to(dev), copy.view(-1,))
        print(f"hello copy shape is {copy.shape}")
    # now set the wgts
    data.edge_attr = torch.where(data.edge_attr.view(-1,) < max_k[data.edge_index[0]].view(-1,),torch.tensor(0.0).to(dev),torch.tensor(1.0).to(dev))
    return

def deg_zero_one(data):
    deg = scatter(data.edge_attr.view(-1,1),data.edge_index[0],dim=0, dim_size=data.x.shape[0], reduce="max")
    tmp_max = deg[data.edge_index[0]]
    deg = scatter(data.edge_attr.view(-1,1),data.edge_index[0],dim=0, dim_size=data.x.shape[0], reduce="min")
    tmp_min = deg[data.edge_index[0]] #(cause edge index is like ([0,0,0],[1,3,23]])
    data.edge_attr = (data.edge_attr - tmp_min.view(-1,))/(tmp_max.view(-1,)-tmp_min.view(-1, ))
    return

# there is topk implementation also!
def wgts_thresh(data,thresh, dev):
    data.edge_attr = torch.where(data.edge_attr < torch.tensor(thresh).to(dev),torch.tensor(0.0).to(dev),torch.tensor(1.0).to(dev))
    # data.edge_attr = torch.where(data.edge_attr > torch.tensor(thresh).to(dev),torch.tensor(0.0).to(dev),torch.tensor(1.0).to(dev))
    # data.edge_attr = torch.where(data.edge_attr >= thresh,1.0,data.edge_attr) 
    return

from torch_geometric.transforms import ToUndirected
def symmetrize(data):
    transform = ToUndirected(reduce="max")
    data = transform(data)
    return

def get_deg(data, dev):
    # data.deg = scatter(data.edge_attr.view(0,1),data.edge_index[0],dim=0, dim_size=data.x.shape[0], reduce="add")
    data.deg = scatter(torch.ones((data.edge_index.shape[1],1)).to(torch.float64).to(dev),data.edge_index[0],dim=0, dim_size=data.x.shape[0], reduce="add")
    # data.edge_attr = torch.where(data.edge_attr >= thresh,1.0,data.edge_attr) 
    return


#import time
#from tqdm import tqdm
#from Jacobi import Jacobi_iterative
#def run_jac(data, dev, itr, k, eps):
#    my_edges = {}
#    # data = data.to(dev)
#    start_time  = time.time() 
#    for i in tqdm(range(data.x.shape[0])):
#        test_tensor = torch.zeros(data.x.shape[0],1)
#        test_tensor[i] = 1#should work!
#        test_tensor = test_tensor.to(dev)
#        hp = dict(dim_size=data.x.shape[0]) #done
#        # if i == 0:
#            # print("hello just before the iteration")
#            # print(test_tensor.view(1,-1))
#            # print(torch.max(test_tensor))
#        j_i = Jacobi_iterative(data, **hp) #done
#        new_sig = j_i.run(itr, test_tensor, node=i)
#        # indices = torch.where(new_sig >= eps)
#        # indices = torch.where((new_sig >= new_sig[i]-0.00000001)&(new_sig <= new_sig[i] + 0.00000001))
#        # indices = torch.where((new_sig >= new_sig[i]-0.00001)&(new_sig <= new_sig[i] + 0.00001))
#        # print("new_sig shape", new_sig.shape)
#        # indices = torch.topk(new_sig, 8, dim=0)
#        #indices = torch.topk(new_sig.view(-1,), 8)
#        indices = torch.topk(new_sig.view(-1,), k)
#        # indices = torch.topk(new_sig.view(-1,), 4)
#        # indices = torch.topk(new_sig.view(-1,), 5)
#        # indices = torch.topk(new_sig.flatten(), 8)
#        # indices = torch.where(new_sig >= new_sig[i])
#        # my_edges[i] = indices[0]
#        my_edges[i] = indices[1]
#        # if i < 10:
#        #     if i == 0:
#                # print(data.edge_index[:,:200])
#                # print(data.edge_attr[:200])
#                # print(new_sig.view(1,-1))
#                # print(torch.max(new_sig))
#            #print(f'my_edges[{i}]: {my_edges[i]}')
#    end_time = time.time()
#    # data = data.to("cpu")
#    # create new edge_index vector.
#    my_edges_i = []
#    my_edges_j = []
#    for i in tqdm(range(data.x.shape[0])):
#        my_edges_j += my_edges[i].tolist()
#        my_edges_i += [i for _ in range(len(my_edges[i]))]
#    my_edge_index = [my_edges_i, my_edges_j]
#    # my_edge_index = [my_edges_j, my_edges_i]
#    data.edge_index = torch.LongTensor(my_edge_index).to(dev)
#    return


def get_mnist_4_9():
    # Define data transformations
    transform = transforms.Compose([ transforms.ToTensor()])

    # Download and load the training and test datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter examples with label 4 from the training dataset
    train_features_label_4 = []
    for data, target in trainset:
        if target == 4:
            train_features_label_4.append(data.view(-1))  # Flatten the image tensor to a 1D tensor
    # Stack the training features into a single tensor
    train_features_label_4 = torch.stack(train_features_label_4)

    # Filter examples with label 4 from the training dataset
    train_features_label_9 = []
    for data, target in trainset:
        if target == 9:
            train_features_label_9.append(data.view(-1))  # Flatten the image tensor to a 1D tensor
    # Stack the training features into a single tensor
    train_features_label_9 = torch.stack(train_features_label_9)

    # Printing the shapes of the resulting tensors
    print("Train Features - Label 4: ", train_features_label_4.shape)
    print("Train Features - Label 4: ", train_features_label_9.shape)

    return train_features_label_4, train_features_label_9
