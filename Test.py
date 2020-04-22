import sys
import numpy as np
import scipy.sparse as sp
# import hypernetx as hnx
#
# # hgr file
# # 1st line :  # of nets, # of cells
# # rest of the lines : Cells that are part of the net. One line per net. Cells are numbered serially.
#
# # Function returns the index of all matches of element "ele" in array 'array'
# def get_index(array,ele):
# 	return [i for i, j in enumerate(array) if j == ele]
#
# # Function to check if there is a common element between two arrays
# def check_if_common_element(arr1,arr2):
# 	a_set=set(arr1)
# 	b_set=set(arr2)
# 	if (a_set & b_set):
# 		return (a_set & b_set).pop()
# 	else:
# 		return -1
#
#
# def HGR2Adj(filename):
# 	'''
# 		Takes a .hgr file and generates a SciPy Adjecency matrix
# 	'''
#
# 	file = open(filename, 'r')
# 	# Storing each line of the file in the list lst[]
# 	lst = []
# 	for line in file:
# 		lst.append(line)
#
# 	hgr_elements = []
# 	for item in lst:
# 		hgr_elements += [item.strip(" \n").split(" ")]
#
# 	# Storing the graph in COO format
# 	data = []
# 	row = []
# 	col = []
#
# 	# hgr file description
# 	no_of_nets = hgr_elements[0][0]
# 	no_of_cells = hgr_elements[0][1]
# 	# Skipping the first line
# 	hgr_elements_iter = iter(hgr_elements)
# 	next(hgr_elements_iter)
# 	for edge in hgr_elements_iter:
# 		no_of_nodes = len(edge)
# 		weight = 1 / (no_of_nodes - 1)  # Weight of the edge in the graph
# 		# print(edge)
# 		# Updating the weights
# 		for i in range(len(edge) - 1):
# 			for j in range(i + 1, len(edge)):
# 				# Check if an edge already exists between i and j
# 				# If it does, then update the weight
# 				if (edge[i] in row) and (edge[j] in col) and (
# 						check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j])) >= 0):
# 					# The edge already exists
# 					data[check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j]))] += weight
# 				else:
# 					data.append(weight)
# 					row.append(edge[i])
# 					col.append(edge[j])
# 	# print(data)
# 	# print(row)
# 	# print(col)
# 	final_row = np.asarray(list(map(int, row + col))) - 1
# 	final_col = np.asarray(list(map(int, col + row))) - 1
# 	final_data = np.asarray(list(map(float, data + data)))
# 	N = int(no_of_cells)
# 	A = sp.csr_matrix((final_data, (final_row, final_col)), shape=(N, N))
# 	return A
#
#
# filename = './test_hyp.hgr'
# # A = HGR2Adj(filename)
#
# scenes = {
#     0: ('FN', 'TH'),
#     1: ('TH', 'JV'),
#     2: ('BM', 'FN', 'JA'),
#     3: ('JV', 'JU', 'CH', 'BM'),
#     4: ('JU', 'CH', 'BR', 'CN', 'CC', 'JV', 'BM'),
#     5: ('TH', 'GP'),
#     6: ('GP', 'MP'),
#     7: ('MA', 'GP')
# }
#
# H = hnx.Hypergraph(scenes)
# hnx.draw(H)
#

# N = 3
# row = np.array([0,0,1,2])
# col = np.array([1,2,0,0])
# data = np.array([2,3,2,3])
# A = sp.csr_matrix((data, (row, col)), shape=(N, N))
# Ad = (A.todense())
# print(Ad)
# node_idx = np.array([0,1,1])
# B = A.tocoo()
# print('hi')

