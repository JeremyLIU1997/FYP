#!/usr/local/bin/python3

# Author: LIU Le
# Student ID: 15103617D

# This file implements the ALS algorithm for
# recommender system. This is a local version

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

def euclidean(a, b):
	return np.linalg.norm(a - b) ** 2

def euclidean_exclude_zero(a,b):
	err = 0
	for i in range(len(a)):
		for j in range(len(a[i])):
			if (a[i][j] == 0 or b[i][j] == 0):
				continue
			err += (a[i][j] - b[i][j])**2
	return err

tempU = 0
tempM = 0
temprecons = 0

def als_fit(mat): # mat is the rating matrix, mat = U * M
	no_nonzero = np.count_nonzero(mat)
	U = np.random.rand(mat.shape[0], Nf).astype(float)
	M = np.random.rand(Nf, mat.shape[1]).astype(float)
	# initialize first row of M with the average rating of the movie
	# other entries a small random number
	M[0,:] = np.sum(R,axis=0) / np.count_nonzero(mat,axis=0)
	N_rows = mat.shape[0]
	N_cols = mat.shape[1]
	lU = np.random.rand(mat.shape[0],Nf).astype(float)
	lM = np.random.rand(Nf,mat.shape[1]).astype(float)
	lam = 0.06
	lamI = np.identity(Nf) * lam
	current_err = 0
	last_err = 0
	print("Number of nonzero: " + str(no_nonzero))
	for current_iter in range(N_iter): # iteration
		print("Iteration #" + str(current_iter+1) + ": ", end="")
		# update each column in M
		for i in range(N_cols):
			users = list(np.nonzero(mat[:,i])[0])
			if len(users) == 0:
				continue
			Um = U[users,:].T
			vector = np.matmul(Um,mat[users,i])
			matrix = np.matmul(Um,Um.T) + np.count_nonzero(mat[:,i]) * lamI
			try:
				lM[:,i] = np.matmul(np.linalg.inv(matrix),vector)
			except Exception:
				print(matrix)
				exit(0)
		M = lM
		tempM = M
		# update each row in U
		for i in range(N_rows):
			movies = list(np.nonzero(mat[i,:])[0])
			if len(movies) == 0:
				continue
			Mm = M[:,movies]
			vector = np.matmul(Mm,mat[i,movies])
			matrix = np.matmul(Mm,Mm.T) + np.count_nonzero(mat[i,:]) * lamI
			try:
				lU[i,:] = np.matmul(np.linalg.inv(matrix),vector)
			except Exception:
				print(matrix)
				exit(0)
		U = lU
		tempU = U

		last_err = current_err
		recons = np.matmul(U,M)
		temprecons = recons
		current_err = euclidean_exclude_zero(mat, recons) / no_nonzero
		# print("U: \n" + str(U))
		# print("M: \n" + str(M))
		print(str(current_err) + "(%" + str(100 * current_err/last_err) + ")")
	print("recons_err: \n" + str(R - recons))
		

if __name__ == '__main__':
	
	# adjustable parameter
	input = "../Data/netflix_data/my_data_30.txt"
	Nf = 2
	N_iter = 5000
	#========================================================
	# CAUTION: be extremely careful when printing a very large
	# array. Tried printing a 191M array, job freezes for minutes
	# with that particular process occupying 100% of allocated
	# CPU time
	# ****** print(data) ******
	print("Loading data...")
	#R = load(input)
	# R = np.random.rand(1000,200).astype(float)
	# R = np.array([[1,3,3,5,3],[4,2,3,3,1],[5,1,5,3,2],[4,2,3,2,4]])
	# R = load(input)
	# R = np.array([[5,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,5,4],[5,3,0,0]])
	R = np.array([[5,3,2,1],[4,5,3,1],[1,1,2,5],[1,3,2,4],[3,1,5,4],[5,3,2,2]])

	print("R.shape: " + str(R.shape))

	model = NMF(n_components=Nf)
	print("Factorizing...")
	print("Nf = " + str(Nf))


	try:
		als_fit(R)
		exit(0)
		print('Sklearn Non-negative matrix factorization')
		model = NMF(n_components=2, init='random', random_state=0)
		W = model.fit_transform(R)
		H = model.components_
		print("Sklearn err: " + str(euclidean_exclude_zero(R,np.matmul(W,H)) / 24))
	except KeyboardInterrupt:
		save(tempU, "../Model/user.txt")
		save(tempM, "../Model/item.txt")
		save(temprecons, "../Model/recons.txt")
		exit(0)
