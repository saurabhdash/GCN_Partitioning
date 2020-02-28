import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H, A):
        W = self.weight
        b = self.bias

        HW = torch.mm(H, W)
        # AHW = SparseMM.apply(A, HW)
        AHW = torch.spmm(A, HW)
        if self.bias is not None:
            return AHW + b
        else:
            return AHW

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


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
    D = torch.sum(A, dim=1)
    Gamma = torch.mm(Y.t(), D.unsqueeze(1))

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
    D = torch.sparse.sum(A, dim=1).to_dense()
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.tensor([0.])
    idx = A._indices()
    for i in range(idx.shape[1]):
        # print(YbyGamma[idx[0,i],:].dtype)
        # print(Y_t[:,idx[1,i]].dtype)
        loss += torch.dot(YbyGamma[idx[0,i],:], Y_t[:,idx[1,i]])
        # print(loss)
    # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)
    return loss

def RandLargeGraph(N):
    i = (torch.LongTensor(2,int(0.05 * N)).random_(0, N))
    v = 1. * torch.ones(int(0.05 * N))
    return torch.sparse.FloatTensor(i, v, torch.Size([N, N]))

class CutLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Y, A):
        ctx.save_for_backward(Y,A)
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        YbyGamma = torch.div(Y, Gamma.t())
        # print(Gamma)
        Y_t = (1 - Y).t()
        loss = torch.tensor([0.], requires_grad=True)
        idx = A._indices()
        data = A._values()
        for i in range(idx.shape[1]):
            # print(YbyGamma[idx[0,i],:].dtype)
            # print(Y_t[:,idx[1,i]].dtype)
            # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
            loss += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i]
            # print(loss)
        # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        Y, A, = ctx.saved_tensors
        idx = A._indices()
        data = A._values()
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        # print(Gamma.shape)
        gradient = torch.zeros_like(Y)
        # print(gradient.shape)
        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                alpha_ind = (idx[0, :] == i).nonzero()
                alpha = idx[1, alpha_ind]
                A_i_alpha = data[alpha_ind]
                temp = A_i_alpha / torch.pow(Gamma[j], 2) * (Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * (
                            Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j])))
                gradient[i, j] = torch.sum(temp)

        # print(gradient)
        return gradient, None

def test_backward(Y,A):
    idx = A._indices()
    data = A._values()
    D = torch.sparse.sum(A, dim=1).to_dense()
    Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    # print(Gamma.shape)
    gradient = torch.zeros_like(Y, requires_grad=True)
    # print(gradient.shape)
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            if i == 1 and j == 0:
                alpha_ind = (idx[0, :] == i).nonzero()
                alpha = idx[1, alpha_ind]
                A_i_alpha = data[alpha_ind]
                temp = A_i_alpha/ torch.pow(Gamma[j], 2) * ( Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * ( Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j]) ) )
                gradient[i, j] = temp


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

def rev_symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, 1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)