#!/usr/local/bin/python3

# Author: LIU Le
# Student ID: 15103617D

# This file implements ALS algorithm for recommender system
# with Spark. The algorithm summary is as follows:

"""
********************************************************************************
ALS with Spark:
1. Broadcast R (by row and by column) with the sparse representation
2. Broadcast U
3. (Update M first) For column in M, map (send to processing nodes to compute)
4. Broadcast M
5. (Update U now)   For row in U, map (send to processing nodes to compute)
6. Repeat 2-4 until convergence
********************************************************************************
"""

# Spark
from pyspark import SparkContext
from pyspark.sql import SparkSession
# from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.linalg import Matrix, Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
from pyspark.mllib.util import MLUtils
# math
import numpy as np
# my modules
from load import *
# others
import os

# global variables
sc = 0
spark = 0

def init():
	global sc
	sc = SparkContext()
	# sc.setLogLevel("ALL")
	sc.addPyFile("/Users/LeLe/Documents/Sem7/FYP/code/NMF/src/load.py")
	spark = SparkSession(sc)
	# clear model from last time
	# os.system("hdfs dfs -rm -R /NMF/model/my_model")
def test_func(row):
	return row + 100

def update_M(col):
	# gather broadcast variables
	R_sparse_column_ = R_sparse_column.value
	column_nonzero_count_ = column_nonzero_count.value
	U_ = U.value
	col_index = int(col)
	users = R_sparse_column_[1][col_index]
	Um = U_[users,:]

	lam = 0.06
	lamI = np.identity(Nf) * lam
	vector = np.matmul(Um.T, extract_from_sparse(R_sparse_column_, [col_index], option = "col")[users])
	matrix = np.matmul(Um.T,Um) + column_nonzero_count_[col_index] * lamI
	return np.matmul(np.linalg.inv(matrix),vector)

def update_U(row):
	# gather broadcast variables
	R_sparse_row_ = R_sparse_row.value
	row_nonzero_count_ = row_nonzero_count.value
	M_ = M.value
	row_index = int(row)
	movies = R_sparse_row_[1][row_index]
	Mm = M_[movies,:]

	lam = 0.06
	lamI = np.identity(Nf) * lam
	vector = np.matmul(Mm.T, extract_from_sparse(R_sparse_row_, [row_index], option = "row")[:,movies].T)
	matrix = np.matmul(Mm.T,Mm) + row_nonzero_count_[row_index] * lamI
	return np.matmul(np.linalg.inv(matrix),vector)

def euclidean_exclude_zero(a,b):
	err = 0
	if (a.shape != b.shape):
		print("Shape incompatible. Exit.")
		exit(1)
	for i in range(len(a)):
		for j in range(len(a[i])):
			if (a[i][j] == 0 or b[i][j] == 0):
				continue
			err += (a[i][j] - b[i][j])**2
	return err

######################################    MAIN FUNCTION    ##############################################

init()

# adjustable parameters
Nf = 5
N_iter = 100

input = "/Users/LeLe/Documents/Sem7/FYP/code/NMF/Data/netflix_data/my_data_10_sorted.txt"
R_sparse_column, R_sparse_row = load_as_sparse(input)
R_height = R_sparse_row[0][0]
R_width = R_sparse_row[0][1]
# calculate cardinalities
row_nonzero_count = column_nonzero_count = []
for i in range(len(R_sparse_column[1])):
	column_nonzero_count.append(len(R_sparse_column[1][i]))
for i in range(len(R_sparse_row[1])):
	row_nonzero_count.append(len(R_sparse_row[1][i]))

# initialize U and M, and append line number to it for
# identification during update process in workers
U = np.random.rand(R_sparse_row[0][0], Nf).astype(float)
M = np.random.rand(Nf, R_sparse_row[0][1]).astype(float)
# initialize first row of M with the average rating of the movie
# other entries a small random number
for i in range(len(R_sparse_column[2])):
	M[0,i] = sum(R_sparse_column[2][i]) / len(R_sparse_column[2][i])
M = M.T # store column as rows, easily for parallelization

# broadcast data to workers
print("Broadcasting...")
R_sparse_row = sc.broadcast(R_sparse_row)
R_sparse_column = sc.broadcast(R_sparse_column)
column_nonzero_count = sc.broadcast(column_nonzero_count)
row_nonzero_count = sc.broadcast(row_nonzero_count)
print("Broadcast success!\n")

R_dense = load(input)
no_nonzero = np.count_nonzero(R_dense)
print("nonzero: " + str(no_nonzero))
last_err = 0
current_err = 0


for i in range(N_iter):
	print("Iteration #" + str(i + 1) + ": ", end = "")
	U = sc.broadcast(np.array(U))

	dummy_M = np.zeros(R_width)
	for i in range(len(dummy_M)):
		dummy_M[i] = i
	dummy_M = sc.parallelize(dummy_M)
	dummy_M = dummy_M.map(update_M).cache()

	M = dummy_M.collect()
	M = sc.broadcast(np.array(M).reshape((R_width,Nf)))
	
	dummy_U = np.zeros(R_height)
	for i in range(len(dummy_U)):
		dummy_U[i] = i
	dummy_U = sc.parallelize(dummy_U, numSlices = 1000)
	dummy_U = dummy_U.map(update_U).cache()

	U = dummy_U.collect()
	M = M.value
	U = np.array(U).reshape((R_height,Nf))
	M = np.array(M)
	last_err = current_err
	current_err = euclidean_exclude_zero(R_dense,np.matmul(U,M.T)) / no_nonzero
	print(str(current_err) + "(%" + str(100 * current_err/last_err) + ")")
	"""
	if current_err > last_err:
		print("Learning complete.")
		break
	"""
