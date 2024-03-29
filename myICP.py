import numpy
import numpy as np
import random
import matplotlib.pyplot as plt


def nearest_point(P, Q):
    P = np.array(P)
    Q = np.array(Q)
    dis = np.zeros(P.shape[0])
    index = np.zeros(Q.shape[0], dtype = np.int)

    for i in range(P.shape[0]):
        minDis = np.inf
        for j in range(Q.shape[0]):
            tmp = np.linalg.norm(P[i] - Q[j], ord = 1)
            if minDis > tmp:
                minDis = tmp
                index[i] = j
        dis[i] = minDis
    return dis, index


def find_optimal_transform(P, Q):
    meanP = np.mean(P, axis = 0)
    meanQ = np.mean(Q, axis = 0)
    P_ = P - meanP
    Q_ = Q - meanQ

    W = np.dot(Q_.T, P_)
    U, S, VT = np.linalg.svd(W)
    R = np.dot(U, VT)
    # if np.linalg.det(R) < 0:
    #    R[2, :] *= -1

    T = meanQ.T #- np.dot(R, meanP.T)
    return U, T


def icp(src, dst, word1, word2, maxIteration=2, tolerance=0.1, controlPoints=10):
    A = np.array(src)
    B = np.array(dst)
    lastErr = 0
    if A.shape[0] != B.shape[0]:
        length = min(A.shape[0], B.shape[0])
        length = min(length, controlPoints)
        sampleA = random.sample(range(A.shape[0]), length)
        sampleB = random.sample(range(B.shape[0]), length)
        P = np.array([A[i] for i in sampleA])
        Q = np.array([B[i] for i in sampleB])
    else:
            P = A
            Q = B

    for i in range(maxIteration):
        dis, index = nearest_point(P, Q)
        R, T = find_optimal_transform(P, Q[index, :])
        A = np.dot(R, A.T).T + np.array([T for j in range(A.shape[0])])
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])

        meanErr = np.sum(dis) / dis.shape[0]
        if abs(lastErr - meanErr) < tolerance:
            break
        lastErr = meanErr

        #visualization
        # ax = plt.subplot(1, 1, 1)
        # ax.scatter(P[:, 0], P[:, 1], c='r')
        # ax.scatter(Q[:, 0], Q[:, 1], c='g')
        # plt.show(block = True)

    print(word1 + " " + word2 + " Iteration : " + str(i) + " with Err : " + str(lastErr))
    R, T = find_optimal_transform(A, np.array(src))
    return lastErr
