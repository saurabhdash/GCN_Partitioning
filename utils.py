import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
# import networkx as nx
from models import *
import time
from numba import njit, jit


def input_matrix():
    '''
    Returns a test sparse SciPy adjecency matrix
    '''
    # N = 8
    # data = np.ones(2 * 11)
    # row = np.array([0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,6,6,6,7,7])
    # col = np.array([1,2,0,2,3,0,1,3,1,2,4,3,5,6,7,4,6,4,5,7,4,6])

    N = 7
    data = np.ones(2 * 9)
    row = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6])
    col = np.array([2, 3, 4, 6, 0, 4, 5, 6, 0, 4, 5, 1, 2, 3, 2, 3, 1, 2])

    # N = 3
    # data = np.array([1/2,1/2,1/3,1/3])
    # row = np.array([0,1,1,2])
    # col = np.array([1,0,2,1])

    A = sp.csr_matrix((data, (row, col)), shape=(N, N))

    return A

def check_grad(model, x, adj, A, As):
    Y = model(x, adj)
    Y.register_hook(print)
    print(Y)
    print('\n')
    loss1 = CutLoss.apply(Y,As)
    loss = custom_loss(Y, A)
    print('\n')
    loss.backward()
    print('\n')
    loss1.backward()
    # test_backward(Y,As)
    # test = torch.autograd.gradcheck(CutLoss.apply, (Y.double(), As.double()), check_sparse_nnz=True)


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


class GCN(torch.nn.Module):

    def __init__(self, gl, ll, dropout):
        super(GCN, self).__init__()
        if ll[0] != gl[-1]:
            assert 'Graph Conv Last layer and Linear first layer sizes dont match'
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.graphlayers = nn.ModuleList([GraphConvolution(gl[i], gl[i+1], bias=True) for i in range(len(gl)-1)])
        self.linlayers = nn.ModuleList([nn.Linear(ll[i], ll[i+1]) for i in range(len(ll)-1)])

    def forward(self, H, A):
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        for idx, hidden in enumerate(self.graphlayers):
            H = F.relu(hidden(H,A))
            if idx < len(self.graphlayers) - 2:
                H = F.dropout(H, self.dropout, training=self.training)

        H_emb = H

        for idx, hidden in enumerate(self.linlayers):
            H = F.relu(hidden(H))
            # if idx < len(self.linlayers) - 1:
            #     H = F.dropout(H, self.dropout, training=self.training)

        # print(H)
        return F.softmax(H, dim=1)

    def __repr__(self):
        return str([self.graphlayers[i] for i in range(len(self.graphlayers))] + [self.linlayers[i] for i in range(len(self.linlayers))])


def custom_loss(Y, A):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : dense adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    D = torch.sum(A, dim=1)
    Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    # print(Gamma)
    loss = torch.sum(torch.mm(torch.div(Y.float(), Gamma.t()), (1 - Y).t().float()) * A.float())
    return loss

def custom_loss_equalpart(Y, A, beta):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : dense adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A + beta* (I^T Y - n/g)^2
    '''
    # beta = 0.0001
    D = torch.sum(A, dim=1)
    n = Y.shape[0]
    g = Y.shape[1]
    Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    # print(F.softmax(Y/0.1, 1)[0:10,:])
    # print(Y[0:10,:])
    # print(torch.mm(torch.ones(1,n).to('cuda')/n, F.softmax(Y/0.1, 1)))
    # balance_loss = torch.sum(torch.pow(torch.mm(torch.ones(1,n).to('cuda')/n, F.softmax((Y + torch.randn(Y.shape).to('cuda') * 0.2)/0.1, 1)) - 1/g , 2))
    balance_loss = torch.sum(torch.pow(torch.mm(torch.ones(1, n).to('cuda') / n, Y) - 1 / g, 2))
    partition_loss = torch.sum(torch.mm(torch.div(Y.float(), Gamma.t()), (1 - Y).t().float()) * A.float())
    print('Partition Loss:{0:.3f} \t Balance Loss:{0:.3f}'.format(partition_loss, balance_loss))
    loss = partition_loss + beta * balance_loss
    return loss


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def custom_loss_sparse(Y, A):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    D = torch.sparse.sum(A, dim=1).to_dense()
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.tensor([0.]).to('cuda')
    idx = A._indices()
    for i in range(idx.shape[1]):
        loss += torch.dot(YbyGamma[idx[0,i],:], Y_t[:,idx[1,i]])
    return loss

def RandLargeGraph(N,c):
    '''
    Creates large random graphs with c fraction connections compared to the actual graph size
    '''
    i = (torch.LongTensor(2,int(c * N)).random_(0, N))
    v = 1. * torch.ones(int(c * N))
    return torch.sparse.FloatTensor(i, v, torch.Size([N, N]))


def test_backward(Y,A):
    '''
    This a function to debug if the gradients from the CutLoss class match the actual gradients
    '''
    idx = A._indices()
    data = A._values()
    D = torch.sparse.sum(A, dim=1).to_dense()
    Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    # print(Gamma.shape)
    gradient = torch.zeros_like(Y, requires_grad=True)
    # print(gradient.shape)
    # print(idx)
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            # if i == 1 and j == 0:
            alpha_ind = (idx[0, :] == i).nonzero()
            alpha = idx[1, alpha_ind]
            A_i_alpha = data[alpha_ind]
            temp = A_i_alpha/ torch.pow(Gamma[j], 2) * ( Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * ( Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j]) ) )
            gradient[i, j] = torch.sum(temp)

            l_idx = list(idx.t())
            l2 = []
            l2_val = []
            # [l2.append(mem) for mem in l_idx if((mem[0] != i).item() and (mem[1] != i).item())]
            for ptr, mem in enumerate(l_idx):
                if ((mem[0] != i).item() and (mem[1] != i).item()):
                    l2.append(mem)
                    l2_val.append(data[ptr])
            extra_gradient = 0
            if(l2 != []):
                for val, mem in zip(l2_val, l2):
                    extra_gradient += (-D[i] * torch.sum(Y[mem[0],j] * (1 - Y[mem[1],j]) / torch.pow(Gamma[j],2))) * val

            gradient[i,j] += extra_gradient

    print(gradient)





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def test_partition(Y):
    _, idx = torch.max(Y, 1)
    return idx

def Train_dense(model, x, adj, A, optimizer):
    '''
    Training Specifications
    '''

    max_epochs = 100
    min_loss = 100
    for epoch in (range(max_epochs)):
        Y = model(x, adj)
        # loss = CutLoss.apply(Y,A)
        loss = custom_loss(Y, A)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "./trial_weights.pt")
        loss.backward()
        optimizer.step()

# Function returns the index of all matches of element "ele" in array 'array'
def get_index(array,ele):
    return [i for i, j in enumerate(array) if j == ele]

# Function to check if there is a common element between two arrays
def check_if_common_element(arr1,arr2):
    a_set=set(arr1)
    b_set=set(arr2)
    if (a_set & b_set):
        return (a_set & b_set).pop()
    else:
        return -1

def HGR2Adj(filename):
    '''
        Takes a .hgr file and generates a SciPy Adjecency matrix
    '''

    file = open(filename, 'r')
    # Storing each line of the file in the list lst[]
    lst = []
    # start = time.time()
    for line in file:
        lst.append(line)

    hgr_elements = []
    for item in lst:
        hgr_elements += [item.strip(" \n").split(" ")]

    # Storing the graph in COO format
    data = []
    row = []
    col = []

    # hgr file description
    no_of_nets = hgr_elements[0][0]
    no_of_cells = hgr_elements[0][1]
    # Skipping the first line
    hgr_elements_iter = iter(hgr_elements)
    next(hgr_elements_iter)
    itr = 0
    for edge in hgr_elements_iter:
        print(itr)
        start = time.time()
        no_of_nodes = len(edge)
        weight = 1 / (no_of_nodes - 1)  # Weight of the edge in the graph
        # print(edge)
        # Updating the weights
        for i in range(len(edge) - 1):
            for j in range(i + 1, len(edge)):
                # Check if an edge already exists between i and j
                # If it does, then update the weight
                if (edge[i] in row) and (edge[j] in col) and (
                        check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j])) >= 0):
                    # The edge already exists
                    data[check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j]))] += weight
                else:
                    data.append(weight)
                    row.append(edge[i])
                    col.append(edge[j])
        print('Elapsed = {}'.format(time.time() - start))
        itr += 1

    # print(data)
    # print(row)
    # print(col)
    final_row = np.asarray(list(map(int, row + col))) - 1
    final_col = np.asarray(list(map(int, col + row))) - 1
    final_data = np.asarray(list(map(float, data + data)))
    N = int(no_of_cells)
    A = sp.csr_matrix((final_data, (final_row, final_col)), shape=(N, N))
    return A


def Train(model, x, adj, A, optimizer):
    '''
    Training Specifications
    '''

    max_epochs = 200
    min_loss = 100
    for epoch in (range(max_epochs)):
        Y = model(x, adj)
        loss = CutLoss.apply(Y,A)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "./trial_weights.pt")
        loss.backward()
        optimizer.step()


def Test(model, x, adj, A, *argv):
    '''
    Test Final Results
    '''
    model.load_state_dict(torch.load("./trial_weights.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    print(node_idx)
    if argv != ():
        if argv[0] == 'debug':
            print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y,A).item()))
    else:
        print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(CutLoss.apply(Y,A).item()))
    return node_idx


def HypEdgCut(filename, node_idx):
    file = open(filename, 'r')

    int_hyedge = 0
    for idx, line in enumerate(file):
        element = line.strip(" \n").split(" ")
        if idx == 0:
            num_hyedges = int(element[0])
            num_nodes = int(element[1])
        else:
            hyedge = np.asarray(list(map(int,element))) - 1
            # print(hyedge)
            node_class = node_idx[hyedge].data.cpu().numpy()
            if not np.all(node_class == node_class[0]):
                # print('cut')
                int_hyedge += 1

    print('Number of Hyper Edges intersected = {}'.format(int_hyedge))

def sparse_test_and_train(model, x, adj, As, optimizer):
    #Train
    Train(model, x, adj, As, optimizer)

    # Test the best partition
    Test(model, x, adj, As)
#
# @jit(parallel=True, nopython=False)
# def loop(hgr_elements, data, row, col):
#     itr = 0
#     for edge in hgr_elements:
#         if itr == 0:
#             print('itr = 0 skipped')
#             itr += 1
#             continue
#         else:
#             # edge = hgr_elements[itr]
#             print(itr)
#             # start = time.time()
#             no_of_nodes = len(edge)
#             weight = 1 / (no_of_nodes - 1)  # Weight of the edge in the graph
#             # print(edge)
#             # Updating the weights
#             for i in range(len(edge) - 1):
#                 for j in range(i + 1, len(edge)):
#                     # Check if an edge already exists between i and j
#                     # If it does, then update the weight
#                     if (edge[i] in row) and (edge[j] in col) and (
#                             check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j])) >= 0):
#                         # The edge already exists
#                         data[check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j]))] += weight
#                     else:
#                         data.append(weight)
#                         row.append(edge[i])
#                         col.append(edge[j])
#             # print('Elapsed = {}'.format(time.time() - start))
#             itr += 1
#
#     return data[1:], row[1:], col[1:]
#
#
# @njit(parallel=True)
# # Function returns the index of all matches of element "ele" in array 'array'
# def get_index(array,ele):
#     return [i for i, j in enumerate(array) if j == ele]
#
#
# @njit(parallel=True)
# # Function to check if there is a common element between two arrays
# def check_if_common_element(arr1,arr2):
#     a_set=set(arr1)
#     b_set=set(arr2)
#     if (a_set & b_set):
#         return (a_set & b_set).pop()
#     else:
#         return -1
#
#
# def HGR2Adj_parallel(filename):
#     '''
#         Takes a .hgr file and generates a SciPy Adjecency matrix
#     '''
#
#     file = open(filename, 'r')
#     # Storing each line of the file in the list lst[]
#     lst = []
#     # start = time.time()
#     for line in file:
#         lst.append(line)
#
#     hgr_elements = []
#     for item in lst:
#         hgr_elements += [item.strip(" \n").split(" ")]
#
#     # Storing the graph in COO format
#     data = ['a']
#     row = ['a']
#     col = ['a']
#
#     # hgr file description
#     no_of_nets = hgr_elements[0][0]
#     no_of_cells = hgr_elements[0][1]
#     # Skipping the first line
#     # hgr_elements_iter = iter(hgr_elements)
#     # next(hgr_elements_iter)
#     data, row, col = loop(hgr_elements, data, row, col)
#
#
#     # print(data)
#     # print(row)
#     # print(col)
#     final_row = np.asarray(list(map(int, row + col))) - 1
#     final_col = np.asarray(list(map(int, col + row))) - 1
#     final_data = np.asarray(list(map(float, data + data)))
#     N = int(no_of_cells)
#     A = sp.csr_matrix((final_data, (final_row, final_col)), shape=(N, N))
#     return A
