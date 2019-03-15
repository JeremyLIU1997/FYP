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
	spark = SparkSession(sc)
	# clear model from last time
	# os.system("hdfs dfs -rm -R /NMF/model/my_model")
def test_func(row):
	return row + 1

def update_M(row):
	# gather broadcast variables
	R_sparse_column = R_sparse_column.value
	column_nonzero_count = column_nonzero_count.value
	U = U.value






	

def main():
	init()

	# adjustable parameters
	Nf = 2

	input = "../Data/netflix_data/my_data_30_sorted.txt"
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
	for i in range(len(U)):
		U[i][-1] = i
	for i in range(len(M)):
		M[-1][i] = i

	# broadcast data to workers
	print("Broadcasting...")
	R_sparse_column = sc.broadcast(R_sparse_column)
	R_sparse_row = sc.broadcast(R_sparse_row)
	column_nonzero_count = sc.broadcast(column_nonzero_count)
	row_nonzero_count = sc.broadcast(row_nonzero_count)
	U = sc.broadcast(U)
	M = sc.broadcast(M)
	print("Broadcast success!\n")
	


	

if __name__ == '__main__':
	main()