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
from models import *


def Train(model, x, adj, A, optimizer):
    '''
    Training Specifications
    '''

    max_epochs = 100
    min_loss = 100
    for epoch in (range(max_epochs)):
        Y = model(x, adj)
        loss = CutLoss.apply(Y,A)
        # loss = custom_loss(Y, A)
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

def main():
    '''
    Adjecency matrix and modifications
    '''
    A = input_matrix()

    # Modifications
    A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to('cuda') # SciPy to Torch sparse
    As = sparse_mx_to_torch_sparse_tensor(A).to('cuda')  # SciPy to sparse Tensor
    A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to('cuda')   # SciPy to Torch Tensor
    print(A)

    '''
    Declare Input Size and Tensor
    '''
    N = A.shape[0]
    d = 512

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

    # check_grad(model, x, adj, A, As)

    #Train
    Train(model, x, adj, As, optimizer)

    # Test the best partition
    Test(model, x, adj, As)

if __name__ == '__main__':
    main()