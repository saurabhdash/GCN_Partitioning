import pickle
import scipy.sparse as sp
import numpy as np
from utils import *


filename = 'biomedP'
A = HGR2Adj('./hgr_files/'+filename+'.hgr')

pickle.dump(A, open('./pkl_files/'+filename+'.pkl', "wb" ))

# A1 = pickle.load( open('./'+filename+'.pkl', "rb" ))
