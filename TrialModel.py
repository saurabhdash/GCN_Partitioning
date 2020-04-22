from __future__ import division
from __future__ import print_function
import torch.optim as optim
from utils import *
from models import *
import pickle


def Train_dense(model, x, adj, A, optimizer, beta0, hyedge_lst, file):
    '''
    Training Specifications
    '''

    max_epochs = 100
    # min_loss = 100
    min_cut = 10000000000
    min_imbalance = 100
    beta = 0
    for epoch in (range(max_epochs)):
        beta = beta + beta0/max_epochs
        Y = model(x, adj)
        loss = custom_loss_equalpart(Y,A, beta)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        loss.backward()
        optimizer.step()
        cut, imbalance = test_epoch(model, x, adj, hyedge_lst)
        # print(min_cut)
        # print(min_imbalance)
        if cut <= min_cut:
            if imbalance < 45:
                min_cut = cut
                # min_imbalance = imbalance
                torch.save(model.state_dict(), "./model_weights/"+file+"_weights.pt")
        print('======================================')


def test_epoch(model, x, adj, hyedge_lst):
    Y = model(x, adj)
    node_idx = test_partition(Y)
    imbalance = get_stats(node_idx, 2)
    cut = hyedge_lst.get_cut(node_idx)
    return cut, imbalance


def Test_dense(model, x, adj, A, As, beta, hyedge_lst, file, *argv):
    '''
     Test Final Results
     '''
    model.load_state_dict(torch.load("./model_weights/"+file+"_weights.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    # print(node_idx)

    print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss_equalpart(Y, A, beta).item()))
    get_stats(node_idx, 2)
    hyedge_lst.get_cut(node_idx)
    get_edgecut(As, node_idx)
    # return node_idx


def dense_test_and_train(model, x, adj, A, As, optimizer, beta, hyedge_lst, file):
    #Train
    Train_dense(model, x, adj, A, optimizer, beta, hyedge_lst, file)

    # Test the best partition
    print('#####Final Best Result#####')
    node_idx = Test_dense(model, x, adj, A, As, beta, hyedge_lst, file)
    return node_idx


def get_stats(node_idx, n_part):
    bucket = torch.zeros(n_part)
    for i in range(n_part):
        # print(i)
        bucket[i] = torch.sum((node_idx == i).int())

    # imbalance = torch.abs(bucket[0]-len(node_idx)/2) * 100/len(node_idx)
    imbalance = torch.sum(torch.pow(bucket/len(node_idx) - 1/n_part,2)) * 100
    print('Total Elements: {} \t Partition 1: {} \t Partition 2: {} \t equal part = {}'.format(len(node_idx), bucket[0], bucket[1], len(node_idx)/2))
    print('Imbalance = {0:.3f}%'.format(imbalance))
    return imbalance


class HypEdgeLst(object):
    def __init__(self, filename):
        file = open(filename, 'r')
        self.hyedge = []
        for idx, line in enumerate(file):
            element = line.strip(" \n").split(" ")
            if idx == 0:
                self.num_hyedges = int(element[0])
                self.num_nodes = int(element[1])
            else:
                hyedge = np.asarray(list(map(int, element))) - 1
                self.hyedge.append(hyedge)

    def get_cut(self, node_idx):
        int_hyedge = 0
        for hyedge in self.hyedge:
            node_class = node_idx[hyedge].data.cpu().numpy()
            if not np.all(node_class == node_class[0]):
                # print('cut')
                int_hyedge += 1

        print('Number of Hyper Edges intersected = {}'.format(int_hyedge))
        return int_hyedge

def get_edgecut(As, node_idx):
    idx = As.coalesce().indices()
    values = As.coalesce().values()
    different_part = (node_idx[idx[0,:]] ^ node_idx[idx[1,:]]).type(torch.cuda.FloatTensor)
    edgecut = torch.sum(different_part * values) / 2
    totalwt = torch.sum(values)/2
    print('Edgecut = {} \t total edge weight = {} \t fraction = {}'.format(edgecut, totalwt, edgecut*100/totalwt))
    # return edgecut, totalwt


def main():
    '''
    Adjecency matrix and modifications
    '''
    # Loading from Pickle file
    # pkl_filename = 'ibm01'
    # pkl_filename = 'industry2'
    pkl_filename = 'fract'
    filename = './hgr_files/'+pkl_filename+'.hgr'
    A = pickle.load( open('./pkl_files/'+pkl_filename+'.pkl', "rb" ))
    hyedge_lst = HypEdgeLst(filename)

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
    # d = 1
    torch.manual_seed(100)
    x = torch.randn(N, d)
    x = x.to('cuda')

    '''
    Model Definition
    '''
    gl = [d, 256, 128, 64]
    ll = [64, 2]
    dropout = 0
    beta = 1
    # beta = 0.0005
    model = GCN(gl, ll, dropout=dropout).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=5e-3, betas=(0.5, 0.99))
    print(model)

    # check_grad(model, x, adj, A, As)

    # sparse_test_and_train(model, x, adj, As, optimizer)
    dense_test_and_train(model, x, adj, A, As, optimizer, beta, hyedge_lst, pkl_filename)
    # node_idx = Test_dense(model, x, adj, As.to_dense(), beta)

    # print('hi')

if __name__ == '__main__':
    main()