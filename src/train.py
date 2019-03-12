#!/usr/local/bin/python3

# This file is the entry point of a central(non-distributed) NMF algorithm
# Author: LIU Le
# 2019/01/11

import numpy as np
import os
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from threading import Thread
from time import sleep
from als import euclidean_exclude_zero

# my modules
from save_model import *
from load import *

np.set_printoptions(threshold=np.nan)

# initialize the matrices
# why initialize weights to be uniform does not work?
"""
V = np.full((M, N), 0.1).astype(float)
W = np.full((M, R), 0.1).astype(float)
H = np.full((R, N), 0.1).astype(float)
"""
# adjustable parameters
n_iter = 1000
Nf = 50


# V = load("../Data/netflix_data/my_data_30.txt")
V = np.array([[1,3,3,5,3],[4,2,3,3,1],[5,1,5,3,2],[4,2,3,2,4]])
W = np.random.rand(V.shape[0], Nf)
H = np.random.rand(Nf, V.shape[1])
one = np.full((V.shape[0], V.shape[1]), 1).astype(float)

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

# training
stime = time.time()
try:
    for i in range(n_iter):
        print("Iteration #" + str(i + 1))
        # what happens when divided by zero?
        H = H * (divideSkipZero(np.matmul(W.T, divideSkipZero(V, np.matmul(W, H))), np.matmul(W.T, one)))
        W = W * (divideSkipZero(np.matmul(divideSkipZero(V, np.matmul(W, H)), H.T), np.matmul(one, H.T)))
        temp = np.matmul(W, H)
        print(euclidean_exclude_zero(V, temp))
except KeyboardInterrupt:
    etime = time.time()
    print("Training time: " + str(etime - stime) + " seconds.")
    print("Saving model...")
    save(H, "../Model/H.txt")
    save(W, "../Model/W.txt")
    exit()

print(np.matmul(W,H))
etime = time.time()
print("Training time: " + str(etime - stime) + " seconds.")


