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

	dm2 = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])
	sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])

	sc.broadcast(sm)
	"""
	data = sc.parallelize([
		LabeledPoint(0.0, SparseVector(3, {0: 8.0, 1: 7.0})),
		LabeledPoint(1.0, SparseVector(3, {1: 9.0, 2: 6.0})),
		LabeledPoint(1.0, [0.0, 9.0, 8.0]),
		LabeledPoint(2.0, [7.0, 9.0, 5.0]),
		LabeledPoint(2.0, [8.0, 7.0, 3.0])])

	print(data.collect())
	"""






if __name__ == '__main__':
	main()