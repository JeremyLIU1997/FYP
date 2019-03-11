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
print("Loading data...")
R = load(input)

# CAUTION: be extremely careful when printing a very large
# array. Tried printing a 191M array, job freezes for minutes
# with that particular process occupying 100% of allocated
# CPU time
# ****** print(data) ******
print("R.shape: " + str(R.shape))
model = NMF(n_components=Nf)
print("Factorizing...")
print("Nf = " + str(Nf))

def euclidean(a, b):
	return np.linalg.norm(a - b) ** 2

def als_fit(mat): # mat is the rating matrix
	U = np.random.rand(mat.shape[0], Nf)
	M = np.random.rand(Nf, mat.shape[1])
	N_rows = mat.shape[0]
	N_cols = mat.shape[1]
	# lM = np.zeros((Nf, mat.shape[1])).astype(float)
	lM = np.random.rand(Nf, mat.shape[1])
	lamI = np.identity(Nf)
	for current_iter in range(N_iter):
		print("Iteration #" + str(current_iter))
		for i in range(N_cols):
			users = list(np.nonzero(mat[:,i])[0])
			Um = U[users,:].T
			vector = np.matmul(Um,mat[users,i])
			matrix = np.matmul(Um,Um.T) + np.count_nonzero(mat[:,i]) * lamI
			lM[:,i] = np.matmul(np.linalg.inv(matrix),vector)

als_fit(R)
exit(0)
try:
	user_dis = als_fit(R)
except Exception:
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
