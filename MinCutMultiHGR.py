from __future__ import division
from __future__ import print_function
import torch.optim as optim
from utils import *
from models import *
import pickle
import matplotlib.pyplot as plt


class Circuit(object):
    def __init__(self, circuit_name, d=1024):
        self.name = circuit_name
        self.filename = './hgr_files/' + self.name + '.hgr'
        A = pickle.load(open('./pkl_files/' + self.name + '.pkl', "rb"))
        self.hyedge_lst = HypEdgeLst(self.filename)

        A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
        norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
        self.adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to('cuda')  # SciPy to Torch sparse
        self.As = sparse_mx_to_torch_sparse_tensor(A).to('cuda')  # SciPy to sparse Tensor
        self.A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to('cuda')  # SciPy to Torch Tensor

        N = A.shape[0]
        # d = 1024
        # torch.manual_seed(100)
        x = torch.randn(N, d)
        self.x = x.to('cuda')

class Solver(object):
    def __init__(self, model, beta, optimizer, trainckt, valckt, testckt):
        self.max_epochs = 250
        self.trainckt = trainckt
        self.valckt = valckt
        self.testckt = testckt
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.2)
        self.beta = beta

    def train(self):
        self.model.train()
        min_Hcut_avg = 100
        self.loss_arr = []
        for epoch in range(self.max_epochs):
            loss = None
            for ckt in self.trainckt:
                print(ckt.name)
                Y_ckt = self.model(ckt.x, ckt.adj)
                loss_ckt = custom_loss_equalpart(Y_ckt, ckt.A, self.beta)
                print('Epoch {}:   Loss_ckt = {}'.format(epoch, loss_ckt.item()))
                if loss is None:
                    loss = loss_ckt
                else:
                    loss += loss_ckt

            self.loss_arr.append(loss.item())
            loss.backward()
            self.optimizer.step()
            print('###### Validation ######')
            Hcut_avg, imblance_avg = self.validate()
            if Hcut_avg < min_Hcut_avg:
                if imblance_avg < 15:
                    min_Hcut_avg = Hcut_avg
                    torch.save(self.model.state_dict(), "./model_weights/MultiCut.pt")
            self.scheduler.step()
            print('======================================')

        plt.plot(self.loss_arr)
        plt.show()

    def validate(self):
        Hcut = []
        imbalance = []
        self.model.eval()
        with torch.no_grad():
            for ckt in (self.trainckt + self.valckt):
                Y_ckt = self.model(ckt.x, ckt.adj)
                node_idx = test_partition(Y_ckt)
                imbalance.append(get_stats(node_idx, 2))
                Hcut.append(ckt.hyedge_lst.get_cut(node_idx)/ckt.hyedge_lst.num_hyedges)
            Hcut_avg = np.mean(Hcut)
            imblance_avg = np.mean(imbalance)
            print('############')
            print('Average Hyperedge cut percentage is {} and average imbalance is {}'.format(Hcut_avg, imblance_avg))
            print('############')
        self.model.train()
        return Hcut_avg, imblance_avg

    def test(self, ckt):
        print('===Testing Circuit {}==='.format(ckt.name))
        self.model.load_state_dict(torch.load("./model_weights/MultiCut.pt"))
        self.model.eval()
        with torch.no_grad():
            Y_ckt = self.model(ckt.x, ckt.adj)

        node_idx = test_partition(Y_ckt)
        imbalance = get_stats(node_idx, 2)
        Hcut = ckt.hyedge_lst.get_cut(node_idx)
        get_edgecut(ckt.As, node_idx)
        print('\n')


def main():
    '''
        Train, Val and Test Circuits
    '''
    torch.manual_seed(100)
    xdim = 1024

    trainckt = [Circuit('p2', xdim), Circuit('industry2', xdim), Circuit('ibm01', xdim)]

    valckt = [Circuit('structP', xdim), Circuit('fract', xdim)]

    testckt = [Circuit('industry3', xdim)]


    '''
    Model Definition
    '''
    gl = [xdim, 256, 16]
    ll = [16, 2]
    dropout = 0.5
    beta = 1
    # beta = 0.0005
    model = GCN(gl, ll, dropout=dropout).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=5e-3, betas=(0.5, 0.99))
    print(model)
    solver = Solver(model, beta, optimizer, trainckt, valckt, testckt)

    start = time.time()
    solver.train()
    end = time.time() - start

    print('Total time taken: {} s'.format(end))

    for ckt in testckt:
        solver.test(ckt)

if __name__ == '__main__':
    main()
