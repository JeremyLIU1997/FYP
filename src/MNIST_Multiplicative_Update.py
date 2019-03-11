# This file is the entry point of a central(non-distributed) NMF algorithm
# Author: LIU Le
# 2019/01/11

import numpy as np
import os
from mlxtend.data import loadlocal_mnist
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from threading import Thread
from time import sleep

np.set_printoptions(threshold=np.nan)


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28))).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    return plt


# read in the data
x, y = loadlocal_mnist(
    images_path='./MNIST_data/t10k-images-idx3-ubyte',
    labels_path='./MNIST_data/t10k-labels-idx1-ubyte')

# matrix dimensions
M = 784
R = 50
N = 1000
model = open("./model.txt", "w+")
print("Training Setup: ")
print("M: " + str(M))
print("R: " + str(R))
print("N: " + str(N))
model.write("Training Setup: ")
model.write("\n")
model.write("M: " + str(M))
model.write("\n")
model.write("R: " + str(R))
model.write("\n")
model.write("N: " + str(N))
model.write("\n")

"""
M = 5
R = 2
N = 4
"""

# visualize the image
# gen_image(x[0]).show()
# gen_image(sample_x[1]).show()

# initialize the matrices
# why initialize weights to be uniform does not work?
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
for i in range(N):
    V[:, i] = x[i]


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
        exit(1)
    result = np.full(a.shape, 0).astype(float)
    distance = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if b[i][j] == 0:
                result[i][j] = a[i][j]
            else:
                result[i][j] = a[i][j] / b[i][j]
    return result


def show_bases(array, max_images=20, title="Figure"):
    # sub-image width and height
    w = 28
    h = 28

    max = array.shape[1]
    if max_images < array.shape[1]:
        max = max_images

    # how many sub-images
    columns = 4
    if max % 4 == 0:
        rows = int(max / 4)
    else:
        rows = int(max / 4) + 1

    fig = plt.figure(figsize=(columns * 2, rows * 2))
    fig.suptitle(title)

    for i in range(1, max + 1):
        img = array[:, i - 1].reshape(28, 28)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    # save the figures using a separate thread
    thread = Thread(target=saveFigFunc, args=(title,))
    thread.start()
    return plt


def saveFigFunc(title):
    plt.savefig("./figures/" + title)


# clear last log first
# os.system("rm -r /Users/LeLe/PycharmProjects/NMF/figures/*")

# training
n_iter = 1000
f = open("./output.txt", "w+")
stime = time.time()
try:
    for i in range(n_iter):
        print("Iteration #" + str(i + 1))
        # what happens when divided by zero?
        H = H * (divideSkipZero(np.matmul(W.T, divideSkipZero(V, np.matmul(W, H))), np.matmul(W.T, one)))
        W = W * (divideSkipZero(np.matmul(divideSkipZero(V, np.matmul(W, H)), H.T), np.matmul(one, H.T)))
        temp = np.matmul(W, H)
        print(euclidean(V, temp))
        # show_bases(W, title=" Iteration " + str(i + 1), max_images=R)
except KeyboardInterrupt:
    etime = time.time()
    model.write("V: \n" + str(V))
    model.write("\n")
    model.write("W: \n" + str(W))
    model.write("\n")
    model.write("H: \n" + str(H))
    model.write("\n")
    print("Training time: " + str(etime - stime) + " seconds.")
    # show_bases(W,max_images=R).show()
    plt.close()
    f.close()
    model.close()
    exit()

model.write("V: \n" + str(V))
model.write("\n")
model.write("W: \n" + str(W))
model.write("\n")
model.write("H: \n" + str(H))
model.write("\n")
print("Training time: " + str(etime - stime) + " seconds.")
f.close()
model.close()
etime = time.time()
print("Training time: " + str(etime - stime) + " seconds.")
# print("Original: \n" + str(V))
# print()
# print("Predicted: \n" + str(np.dot(W, H)))
