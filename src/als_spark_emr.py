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

base_path = "/home/hadoop"

def init():
	global sc
	sc = SparkContext()
	sc.setLogLevel("ALL")
	# sc.addPyFile("load.py")
	spark = SparkSession(sc)
	# clear model from last time
	# os.system("hdfs dfs -rm -R /NMF/model/my_model")
def test_func(row):
	return row + 1

def update_M(col):
	# gather broadcast variables
	R_sparse_column_ = R_sparse_column.value
	column_nonzero_count_ = column_nonzero_count.value
	U_ = U.value
	U_ = U_[:,0:len(U_[0]) - 1]
	col_index = int(col[-1])
	users = R_sparse_column_[1][col_index]
	Um = U_[users,:]

	lam = 0.06
	lamI = np.identity(Nf) * lam

	vector = np.matmul(Um.T, extract_from_sparse(R_sparse_column_, [col_index], option = "col")[users])
	matrix = np.matmul(Um.T,Um) + column_nonzero_count_[col_index] * lamI
	return np.append(np.matmul(np.linalg.inv(matrix),vector),col_index)

def update_U(row):
	# gather broadcast variables
	R_sparse_row_ = R_sparse_row.value
	row_nonzero_count_ = row_nonzero_count.value
	M_ = M.value
	M_ = M_[:,0:len(M_[0])-1]
	row_index = int(row[-1])
	movies = R_sparse_row_[1][row_index]
	Mm = M_[movies,:]

	lam = 0.06
	lamI = np.identity(Nf) * lam

	vector = np.matmul(Mm.T, extract_from_sparse(R_sparse_row_, [row_index], option = "row")[:,movies].T)
	matrix = np.matmul(Mm.T,Mm) + row_nonzero_count_[row_index] * lamI
	return np.append(np.matmul(np.linalg.inv(matrix),vector),row_index)

def euclidean_exclude_zero(a,b):
	err = 0
	for i in range(len(a)):
		for j in range(len(a[i])):
			if (a[i][j] == 0 or b[i][j] == 0):
				continue
			err += (a[i][j] - b[i][j])**2
	return err

######################################    MAIN FUNCTION    ##############################################

init()

# adjustable parameters
Nf = 2
N_iter = 100

input = "my_data_30_sorted.txt"
R_sparse_column, R_sparse_row = load_as_sparse(input)
row_nonzero_count = column_nonzero_count = []
for i in range(len(R_sparse_column[1])):
	column_nonzero_count.append(len(R_sparse_column[1][i]))
for i in range(len(R_sparse_row[1])):
	row_nonzero_count.append(len(R_sparse_row[1][i]))

# initialize U and M, and append line number to it for
# identification during update process in workers
U = np.random.rand(R_sparse_row[0][0], Nf + 1).astype(float)
M = np.random.rand(Nf + 1, R_sparse_row[0][1]).astype(float)
# initialize first row of M with the average rating of the movie
# other entries a small random number
for i in range(len(R_sparse_column[2])):
	M[0,i] = sum(R_sparse_column[2][i]) / len(R_sparse_column[2][i])
for i in range(len(U)):
	U[i][-1] = i
for i in range(M.shape[1]):
	M[-1][i] = i
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
last_err = 0
current_err = 0
for i in range(N_iter):
	print("Iteration #" + str(i + 1) + ": ", end="")
	U = sc.broadcast(np.array(U))
	# print("1")
	M = sc.parallelize(M)
	# print("2")
	M = M.map(update_M)
	M.cache()
	# print("3")
	M = M.collect()
	# print("4")
	M = sc.broadcast(np.array(M))
	# print("5")
	U = sc.parallelize(U.value,numSlices=100)
	# print("6")
	U = U.map(update_U)
	U.cache()
	# print("7")
	U = U.collect()
	# print("8")
	M = M.value
	# print("9")
	U = np.array(U)
	M = np.array(M)
	last_err = current_err
	current_err = euclidean_exclude_zero(R_dense,np.matmul(U[:,0:len(U[0])-1],M[:,0:len(M[0])-1].T)) / no_nonzero
	print(str(current_err) + "(%" + str(100 * current_err/last_err) + ")")
	"""
	if current_err > last_err:
		print("Learning complete.")
		break
	"""


print(M.take(1))
