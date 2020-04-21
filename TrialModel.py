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
import pickle

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


def Train_dense(model, x, adj, A, optimizer, beta):
    '''
    Training Specifications
    '''

    max_epochs = 1000
    min_loss = 100
    for epoch in (range(max_epochs)):
        Y = model(x, adj)
        loss = custom_loss_equalpart(Y,A, beta)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "./trial_weights.pt")
        loss.backward()
        optimizer.step()

def Test_dense(model, x, adj, A, beta, *argv):
    '''
     Test Final Results
     '''
    model.load_state_dict(torch.load("./trial_weights.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    # print(node_idx)

    print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss_equalpart(Y, A, beta).item()))

    return node_idx


def sparse_test_and_train(model, x, adj, As, optimizer):
    #Train
    Train(model, x, adj, As, optimizer)

    # Test the best partition
    Test(model, x, adj, As)

def dense_test_and_train(model, x, adj, As, optimizer, beta):
    #Train
    Train_dense(model, x, adj, As, optimizer, beta)

    # Test the best partition
    node_idx = Test_dense(model, x, adj, As, beta)
    return node_idx

def get_stats(node_idx, n_part):
    bucket = torch.zeros(n_part)
    for i in range(n_part):
        # print(i)
        bucket[i] = torch.sum((node_idx == i).int())

    print('Total Elements: {} \t Partition 1: {} \t Partition 2: {} \t equal part = {}'.format(len(node_idx), bucket[0], bucket[1], len(node_idx)/2))
    print('Imbalance = {0:.3f}%'.format(torch.abs(bucket[0]-len(node_idx)/2) * 100/len(node_idx)))


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
            # print(node_idx[hyedge])
            if np.logical_xor.reduce(node_idx[hyedge].data.cpu()) != 0:
                # print('cut')
                int_hyedge += 1

    print('Number of Hyper Edges intersected = {}'.format(int_hyedge))




def main():
    '''
    Adjecency matrix and modifications
    '''
    # A = input_matrix()
    # filename = './test_hyp.hgr'
    # filename = './fract.hgr'
    # filename = './industry2.hgr'
    # A = HGR2Adj(filename)

    # Loading from Pickle file
    # pkl_filename = 'industry2'
    pkl_filename = 'fract'
    filename = './'+pkl_filename+'.hgr'
    A = pickle.load( open('./'+pkl_filename+'.pkl', "rb" ))

    # Modifications
    A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to('cuda') # SciPy to Torch sparse
    As = sparse_mx_to_torch_sparse_tensor(A).to('cuda')  # SciPy to sparse Tensor
    A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to('cuda')   # SciPy to Torch Tensor

    # print(A)

    '''
    Declare Input Size and Tensor
    '''
    N = A.shape[0]
    d = 1024

    torch.manual_seed(100)
    x = torch.randn(N, d)
    x = x.to('cuda')

    '''
    Model Definition
    '''
    gl = [d, 64, 16]
    ll = [16, 2]
    dropout = 0
    beta = 0
    # beta = 0.0005
    model = GCN(gl, ll, dropout=dropout).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
    print(model)

    # check_grad(model, x, adj, A, As)

    # sparse_test_and_train(model, x, adj, As, optimizer)
    node_idx = dense_test_and_train(model, x, adj, As.to_dense(), optimizer, beta)
    # node_idx = Test_dense(model, x, adj, As.to_dense(), beta)
    get_stats(node_idx, 2)
    HypEdgCut(filename, node_idx)
    # print('hi')

if __name__ == '__main__':
    main()