from __future__ import division
from __future__ import print_function
import torch.optim as optim
from utils import *
from models import *
import pickle
import argparse
import matplotlib.pyplot as plt
from hgr2sp import HGR2Adj

def Train_dense(model, x, adj, A, As, optimizer, beta0, hyedge_lst, args):
    '''
    Training Specifications
    '''

    max_epochs = 100
    # min_loss = 100
    min_cut = 10000000000
    min_imbalance = 100
    min_edge_cut = 100
    # beta = 0
    Hcut_arr = []
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.2)
    for epoch in (range(max_epochs)):
        # beta = beta + beta0/max_epochs
        Y = model(x, adj)
        loss = custom_loss_equalpart(Y, A, beta0)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        loss.backward()
        optimizer.step()
        cut, imbalance, edge_cut = test_epoch(model, x, adj, hyedge_lst, As, args)
        Hcut_arr.append(cut/hyedge_lst.num_hyedges)
        # print(min_cut)
        # print(min_imbalance)
        if cut <= min_cut:
            if args.parts == 2:
                if imbalance < 15:
                    min_cut = cut
                    # min_imbalance = imbalance
                    torch.save(model.state_dict(), "./model_weights/"+args.circuit+'_'+str(args.parts)+"_MinHedgeCut.pt")
            else:
                min_cut = cut
                # min_imbalance = imbalance
                torch.save(model.state_dict(),
                           "./model_weights/" + args.circuit + '_' + str(args.parts) + "_MinHedgeCut.pt")

        scheduler.step()
        print('======================================')
    plt.plot(Hcut_arr)
    plt.ylabel('Fraction of HyperEdge Cut')
    plt.xlabel('Epochs')
    plt.title(args.circuit)
    plt.show()

def test_epoch(model, x, adj, hyedge_lst, As, args):
    Y = model(x, adj)
    node_idx = test_partition(Y)
    imbalance = get_stats(node_idx, args.parts)
    cut = hyedge_lst.get_cut(node_idx)
    edge_cut = get_edgecut(As, node_idx)
    return cut, imbalance, edge_cut


def Test_dense(model, x, adj, A, As, beta, hyedge_lst, args, *argv):
    '''
     Test Final Results
     '''
    print('===== Hyper Edge Cut Minimization =====')
    model.load_state_dict(torch.load("./model_weights/"+args.circuit+'_'+str(args.parts)+"_MinHedgeCut.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    get_stats(node_idx, args.parts)
    hyedge_lst.get_cut(node_idx)
    get_edgecut(As, node_idx)


def dense_test_and_train(model, x, adj, A, As, optimizer, beta, hyedge_lst, args):
    #Train
    Train_dense(model, x, adj, A, As, optimizer, beta, hyedge_lst, args)

    # Test the best partition
    print('#####Best Result#####')
    node_idx = Test_dense(model, x, adj, A, As, beta, hyedge_lst, args)
    return node_idx

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        dest='mode', default='old', help='To pickle a new hgr file')
    parser.add_argument('--circuit', type=str,
                        dest='circuit', default='fract', help='which circuit to partition')
    parser.add_argument('--beta', default=1, type=float,
                        dest='beta', help='weight assigned to balance')
    parser.add_argument('--parts', default=2, type=int,
                        dest='parts', help='Number of partitions to be divided into')
    return parser.parse_args()

def main():

    '''
    Adjecency matrix and modifications
    '''
    args = parse()
    # Loading from Pickle file
    # pkl_filename = 'structP'
    # pkl_filename = 'ibm01'
    # pkl_filename = 'industry3'
    # pkl_filename = 'fract'
    if args.mode == 'old':
        filename = './hgr_files/'+args.circuit+'.hgr'
        A = pickle.load( open('./pkl_files/'+args.circuit+'.pkl', "rb" ))
    elif args.mode == 'new':
        filename = './hgr_files/' + args.circuit + '.hgr'
        A = HGR2Adj('./hgr_files/' + args.circuit + '.hgr')
    else:
        NotImplementedError()

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
    torch.manual_seed(100)
    x = torch.randn(N, d)
    x = x.to('cuda')

    '''
    Model Definition
    '''
    gl = [d, 256, 16]
    ll = [16, args.parts]
    dropout = 0
    beta = args.beta
    # beta = 0.0005
    model = GCN(gl, ll, dropout=dropout).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=5e-3, betas=(0.5, 0.99))
    print(model)

    start = time.time()
    dense_test_and_train(model, x, adj, A, As, optimizer, beta, hyedge_lst, args)
    end = time.time() - start
    print('Total time taken: {} s'.format(end))

if __name__ == '__main__':
    main()
