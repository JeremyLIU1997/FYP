#!/usr/local/bin/python3
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
from pyspark.mllib.util import MLUtils


sc = SparkContext()
spark = SparkSession(sc)

# Create an RDD of sub-matrix blocks.
blocks = sc.parallelize([((0, 0), Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])),
                         ((1, 0), Matrices.dense(3, 2, [7, 8, 9, 10, 11, 12]))])

# Create a BlockMatrix from an RDD of sub-matrix blocks.
mat = BlockMatrix(blocks, 3, 2)
# Get its size.
m = mat.numRows()  # 6
n = mat.numCols()  # 2
print("m: " + str(m))
print("n: " + str(n))

print(mat)

# Get the blocks as an RDD of sub-matrix blocks.
blocksRDD = mat.blocks

# Convert to a LocalMatrix.
localMat = mat.toLocalMatrix()

# Convert to an IndexedRowMatrix.
indexedRowMat = mat.toIndexedRowMatrix()

# Convert to a CoordinateMatrix.
coordinateMat = mat.toCoordinateMatrix()











"""
R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
     [5,3,0,0]
    ]


with open("./als_small_test.txt","w+") as f:
	for i in range(len(R)):
		for j in range(len(R[i])):
			f.write(str(i)+","+str(j)+","+str(R[i][j])+ "\n")
            """