import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from models import *

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

# loss = custom_loss(Y, A)
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
    loss = torch.tensor([0.])
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
