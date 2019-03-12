#!/usr/local/bin/python3
# Matrix factorization and non-negative matrix factorization
# Most of the following codes are written by Albert Au Yeung
# http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
# Example usage:
#   python3 matrix_factor.py
###############################################################################

import numpy as np
from sklearn.decomposition import NMF
from als import euclidean_exclude_zero
from load import load

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    @INPUT:
        R     : a matrix to be factorized, dimension N x M
        P     : an initial matrix of dimension N x K
        Q     : an initial matrix of dimension M x K
        K     : the number of latent features
        steps : the maximum number of steps to perform the optimisation
        alpha : the learning rate
        beta  : the regularization parameter
    @OUTPUT:
        the final matrices P and Q
    """
    no_nonzero = np.count_nonzero(R)
    err = 0
    Q = Q.T
    last_err = 0
    err = 0
    for step in range(steps):
        print("Iteration #" + str(step+1) + ": ", end="")
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        #eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        last_err = err
        recons = np.matmul(P,Q)
        err = euclidean_exclude_zero(R, recons) / no_nonzero
        print(str(err) + " (%" + str(100 * err/last_err) + ")")
        print(recons)
        """
                if e < 0.001:
            break
        """
    return P, Q.T

###############################################################################

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
         [5,3,0,0]
        ]

    R = np.array(R)
    input = "../Data/netflix_data/my_data_30.txt"
    # R = load(input)
    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    print('Simple matrix factorization')
    print("###############################")
    P, Q = matrix_factorization(R, P, Q, K)
    print(P)
    print(Q)
    print(P.dot(Q.T))

    exit(0)
    print('Non-negative matrix factorization')
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(R)
    H = model.components_
    print(W)
    print(H.T)
    print(W.dot(H))
