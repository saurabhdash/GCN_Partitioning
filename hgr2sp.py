import pickle
import scipy.sparse as sp
import numpy as np
import time

# Function returns the index of all matches of element "ele" in array 'array'
def get_index(array,ele):
    return [i for i, j in enumerate(array) if j == ele]

# Function to check if there is a common element between two arrays
def check_if_common_element(arr1,arr2):
    a_set=set(arr1)
    b_set=set(arr2)
    if (a_set & b_set):
        return (a_set & b_set).pop()
    else:
        return -1

def HGR2Adj(filename):
    '''
        Takes a .hgr file and generates a SciPy Adjecency matrix
    '''

    file = open(filename, 'r')
    # Storing each line of the file in the list lst[]
    lst = []
    # start = time.time()
    for line in file:
        lst.append(line)

    hgr_elements = []
    for item in lst:
        hgr_elements += [item.strip(" \n").split(" ")]

    # Storing the graph in COO format
    data = []
    row = []
    col = []

    # hgr file description
    no_of_nets = hgr_elements[0][0]
    no_of_cells = hgr_elements[0][1]
    # Skipping the first line
    hgr_elements_iter = iter(hgr_elements)
    next(hgr_elements_iter)
    itr = 0
    for edge in hgr_elements_iter:
        print(itr)
        start = time.time()
        no_of_nodes = len(edge)
        weight = 1 / (no_of_nodes - 1)  # Weight of the edge in the graph
        # print(edge)
        # Updating the weights
        for i in range(len(edge) - 1):
            for j in range(i + 1, len(edge)):
                # Check if an edge already exists between i and j
                # If it does, then update the weight
                if (edge[i] in row) and (edge[j] in col) and (
                        check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j])) >= 0):
                    # The edge already exists
                    data[check_if_common_element(get_index(row, edge[i]), get_index(col, edge[j]))] += weight
                else:
                    data.append(weight)
                    row.append(edge[i])
                    col.append(edge[j])
        print('Elapsed = {}'.format(time.time() - start))
        itr += 1

    # print(data)
    # print(row)
    # print(col)
    final_row = np.asarray(list(map(int, row + col))) - 1
    final_col = np.asarray(list(map(int, col + row))) - 1
    final_data = np.asarray(list(map(float, data + data)))
    N = int(no_of_cells)
    A = sp.csr_matrix((final_data, (final_row, final_col)), shape=(N, N))
    return A

def main():
    filename = 'fract'
    A = HGR2Adj('./hgr_files/'+filename+'.hgr')

    pickle.dump(A, open('./pkl_files/'+filename+'.pkl', "wb" ))

    # A1 = pickle.load( open('./'+filename+'.pkl', "rb" ))

if __name__ == '__main__':
    main()