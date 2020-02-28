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

'''
Declare Input Size and Tensor
'''
N = 7
d = 128
x = torch.randn(N,d)
x = x.to('cuda')

'''
Adjecency matrix and modifications
'''

# data = np.ones(2 * 11)
# row = np.array([0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,6,6,6,7,7])
# col = np.array([1,2,0,2,3,0,1,3,1,2,4,3,5,6,7,4,6,4,5,7,4,6])

data = np.ones(2 * 9)
row = np.array([0,0,1,1,2,2,2,2,3,3,3,4,4,4,5,5,6,6])
col = np.array([2,3,4,6,0,4,5,6,0,4,5,1,2,3,2,3,1,2])


# A = torch.tensor([[0,1.,0,1.,0,0],[1.,0,1.,1.,1.,0],[0,1,0,0,1,0],[1,1,0,0,0,1],[0,1,1,0,0,1],[0,0,0,1,1,0]]).to('cuda')
# adj = A + torch.eye(N).to('cuda')
# A = np.array([[0,1.,0,1.,0,0],[1.,0,1.,1.,1.,0],[0,1,0,0,1,0],[1,1,0,0,0,1],[0,1,1,0,0,1],[0,0,0,1,1,0]], dtype='f')
# adj_sp = (sparse.csr_matrix(A + sp.eye(A.shape[0])))
# norm_adj = symnormalise(adj_sp)

A = sp.csr_matrix((data, (row, col)), shape=(N, N))
A_mod = A + sp.eye(A.shape[0])
norm_adj = symnormalise(A_mod)
adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to('cuda')
A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to('cuda')
print(A)

'''
Model Definition
'''
gl = [d, 64, 16]
ll = [16, 2]

model = GCN(gl, ll, 0).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
print(model)

'''
Training Specifications
'''

max_epochs = 2000
min_loss = 100
for epoch in (range(max_epochs)):
    Y = model(x, adj)
    # loss = CutLoss.apply(Y,A)
    loss = custom_loss(Y,A)
    print(loss.item())
    if loss < min_loss:
        torch.save(model.state_dict(), "./trial_weights.pt")
    loss.backward()
    optimizer.step()

# test = torch.autograd.gradcheck(CutLoss.apply, (Y.double(), A.double()), check_sparse_nnz=True)

'''
Test Final Results
'''
model.load_state_dict(torch.load("./trial_weights.pt"))
Y = model(x, adj)
node_idx = test_partition(Y)
print(node_idx)

