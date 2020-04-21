import pickle
import scipy.sparse as sp
import numpy as np
from utils import *

filename = 'industry2'
A = HGR2Adj('./'+filename+'.hgr')

pickle.dump(A, open('./'+filename+'.pkl', "wb" ))

# A1 = pickle.load( open('./'+filename+'.pkl', "rb" ))
