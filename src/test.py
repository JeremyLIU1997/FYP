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

#========================================================
print("Loading data...")
R = load(input)
Nf = 50
# CAUTION: be extremely careful when printing a very large
# array. Tried printing a 191M array, job freezes for minutes
# with that particular process occupying 100% of allocated
# CPU time
# ****** print(data) ******
print("R.shape: " + str(R.shape))
model = NMF(n_components=Nf)
print("Factorizing...")
print("Nf = " + str(Nf))
try:
	user_dis = model.fit_transform(R)
except Exception:
	item_dis = model.components_
	save(user_dis, "../Model/user.txt")
	save(item_dis, "../Model/item.txt")
	print(model.reconstruction_err_)
	exit(0)
item_dis = model.components_
save(user_dis, "../Model/user.txt")
save(item_dis, "../Model/item.txt")
print(model.reconstruction_err_)
recons = np.matmul(user_dis, item_dis)
print("err: " + str(euclidean(R, recons)))

def euclidean(a, b):
	return np.linalg.norm(a - b) ** 2