# This file is the entry point of a central(non-distributed) NMF algorithm
# Author: LIU Le
# 2019/01/11

import numpy as np
from mlxtend.data import loadlocal_mnist
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

np.set_printoptions(threshold=np.nan)


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28))).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest',cmap='gray')
    return plt


x, y = loadlocal_mnist(
    images_path='./MNIST_data/t10k-images-idx3-ubyte',
    labels_path='./MNIST_data/t10k-labels-idx1-ubyte')

# matrix dimensions
M = 784
R = 100
N = 1000

M = 5
R = 3
N = 4

# visualize the image
# gen_image(x[0]).show()
# gen_image(sample_x[1]).show()

# initialize the matrices
"""
V = np.full((M, N), 0.1).astype(float)
W = np.full((M, R), 0.1).astype(float)
H = np.full((R, N), 0.1).astype(float)
"""
V = np.random.rand(M, N)
W = np.random.rand(M, R)
H = np.random.rand(R, N)
one = np.full((M, N), 1).astype(float)

# fill the initialized matrix with data
"""
for i in range(N):
    V[:, i] = x[i]
"""

testV = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]]).astype(float)


def euclidean(a, b):
    """
    if a.shape != b.shape:
        print("Matrix shape not compatible")
        exit(0)
    distance = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            distance = distance + pow(a[i][j] - b[i][j], 2)
            # print(str(a[i][j]) + " " + str(b[i][j]) + " " + str((a[i][j] - b[i][j]) * (a[i][j] - b[i][j])))
    """
    return np.linalg.norm(a - b) ** 2


def KL(a, b):
    if a.shape != b.shape:
        print("Matrix shape not compatible")
        exit(0)
    distance = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if (b[i][j] != 0):
                distance = distance + a[i][j] * np.log(a[i][j] / b[i][j]) - a[i][j] + b[i][j]
                # print(distance)
    return distance


def divideSkipZero(a, b):
    if a.shape != b.shape:
        print("Matrix shape not compatible")
        exit(0)
    result = np.full(a.shape, 0).astype(float)
    distance = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if b[i][j] == 0:
                result[i][j] = a[i][j]
            else:
                result[i][j] = a[i][j] / b[i][j]
                # print(type(a[i][j]))
                # print(str(a[i][j]) + " " + str(b[i][j]) + " " + str(result[i][j]))
    return result


# training
f = open("./output.txt", "w+")
for i in range(1000):
    print()
    print("Iteration #" + str(i + 1))
    """
    print("np.dot(W, H): \n" + str(temp))
    """
    # temp = np.dot(W.T, V)
    # what happens when divided by zero?
    # print("H: \n" + str(H))
    H = H * (divideSkipZero(np.dot(W.T, divideSkipZero(testV, np.dot(W, H))), np.dot(W.T, one)))
    # print("H2: \n" + str(H))
    # f.write("H: \n" + str(H))
    # print("W: \n" + str(W))
    W = W * (divideSkipZero(np.dot(divideSkipZero(testV, np.dot(W, H)), H.T), np.dot(one, H.T)))
    # print("W2: \n" + str(W))
    # f.write("W: \n" + str(W))
    temp = np.dot(W, H)
    print(euclidean(testV, temp))

f.close()
temp = np.dot(W, H)
print("Original: \n" + str(testV))
print()
print("Predicted: \n" + str(temp))
print()
print("Difference: ")
difference = abs((testV - temp))
difference[difference < 0.1] = 0
print(difference)
print()
print(euclidean(testV, temp))
