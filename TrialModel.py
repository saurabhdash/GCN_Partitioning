from __future__ import division
from __future__ import print_function
import tqdm
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from scipy import sparse

def Train(model, x, adj, A, optimizer):
    '''
    Training Specifications
    '''

    max_epochs = 2000
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

    # test = torch.autograd.gradcheck(CutLoss.apply, (Y.double(), A.double()), check_sparse_nnz=True)

def Test(model, x, adj, A):
    '''
    Test Final Results
    '''
    model.load_state_dict(torch.load("./trial_weights.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    print(node_idx)
    print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y,A).item()))

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

    A = sp.csr_matrix((data, (row, col)), shape=(N, N))

    return A


def main():
    '''
    Adjecency matrix and modifications
    '''
    A = input_matrix()

    # Modifications
    A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to('cuda') # SciPy to Torch sparse
    A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to('cuda')   # SciPy to Torch sparse
    print(A)

    '''
    Declare Input Size and Tensor
    '''
    N = A.shape[0]
    d = 128

    torch.manual_seed(100)
    x = torch.randn(N, d)
    x = x.to('cuda')

    '''
    Model Definition
    '''
    gl = [d, 64, 16]
    ll = [16, 2]

    model = GCN(gl, ll, dropout=0.5).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
    print(model)

    Train(model, x, adj, A, optimizer)

    Test(model, x, adj, A)

if __name__ == '__main__':
    main()