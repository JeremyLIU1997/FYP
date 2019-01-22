import numpy

def euclidean(a, b):
    if a.shape != b.shape:
        print("Matrix shape not compatible")
        exit(0)
    distance = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            distance = distance + pow(a[i][j] - b[i][j], 2)
            # print(str(a[i][j]) + " " + str(b[i][j]) + " " + str((a[i][j] - b[i][j]) * (a[i][j] - b[i][j])))
    return distance


def matrix_factorisation(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


R = [
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
]

R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 2

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorisation(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)

print(R)
print(nR)
print(euclidean(R,nR))
