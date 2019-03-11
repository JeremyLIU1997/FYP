#!/usr/local/bin/python3

# imports
from tqdm import tqdm
import sys
import numpy as np
from sklearn.decomposition import NMF

# my modules
from load import load
from save_model import *

# settings
np.set_printoptions(threshold=np.nan)

# adjustable parameters
input = "../Data/netflix_data/my_data_30.txt"
Nf = 50
N_iter = 1000
#========================================================


a = np.array([[1,2,3],[4,5,6],[7,8,9]])

b = a[[0,1],:]
print(b)



