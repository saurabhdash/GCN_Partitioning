import networkx as nx
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt
from utils import *
from networkx.algorithms import community


filename = './fract.hgr'
file = open(filename, 'r')

scenes = {}
for idx, line in enumerate(file):
    element = line.strip(" \n").split(" ")
    if idx == 0:
        H_edges = int(element[0])
        nodes = int(element[1])
    else:
        scenes[idx-1] = set(element)

print(scenes)

def HypEdgCut(filename, node_idx):
    file = open(filename, 'r')

    scenes = {}
    for idx, line in enumerate(file):
        element = line.strip(" \n").split(" ")
        if idx == 0:
            H_edges = int(element[0])
            nodes = int(element[1])
        else:
            scenes[idx - 1] = set(element)

    print(scenes)

#
# H = hnx.Hypergraph(scenes)
# hnx.drawing.draw(H)
# plt.show()

# A = HGR2Adj(filename)
# G = nx.from_scipy_sparse_matrix(A)
# kl_part = community.kernighan_lin_bisection(G, partition=None, max_iter=10, weight='weight', seed=None)
# metis_part = community.nxmetis.partition(G, nparts=2, node_weight='weight', node_size='size', edge_weight='weight', tpwgts=None, ubvec=None, options=None, recursive=False)

# nx.draw(G)
# plt.draw()
# plt.show()
print('hi')

