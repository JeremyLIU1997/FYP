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

def main():
	init()

	input = "../Data/netflix_data/my_data_30_sorted.txt"
	R_sparse_column, R_sparse_row = load_as_sparse(input)

	sc.broadcast(R_sparse_column)

	print("Broadcast success!")
	
	

if __name__ == '__main__':
	main()