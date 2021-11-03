import numpy as np
import numpy.linalg as la
import scipy as sp
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import networkx as nx
# from community import community_louvain
import scipy.sparse.linalg
from sklearn.cluster import KMeans


def partToList(part):
    '''Print partition as a list'''
    N=0
    for b in range(len(part)):
        N=N+len(part[b])
    listP=np.zeros(N)
    for b in range(len(part)):
        for n in part[b]:
            listP[n]=b
    return(listP)

def partToList_for_strings(part,G):
    '''Print partition as a list'''
    N=0
    for b in range(len(part)):
        N=N+len(part[b])
    listP=np.zeros(N)
    for b in range(len(part)):
        for n in part[b]:
            #print('n',n)
            #print('type part=',type(part))
            #ind_n = part[b].index(n)
            ind_n = list(G.nodes).index(n)
            #print('ind_n',ind_n)
            listP[ind_n]=b
    return(listP)


# creates similarity graph by only allowing weights larger than threshold
def similarity_graph(S, threshold):
    G = nx.empty_graph()
    n = len(S)
    # upper half S
    for i in range(n):
        for j in range(i + 1, n):
            if abs(S[i][j]) > threshold:
                G.add_edge(i, j, weight=S[i][j])
    return G


# takes only real parts of the eigenvector (in acse the matrix is not symmetric) and does transpose
def real_Vec(Vec):
    RVec = []
    for v in Vec:
        v = [e.real for e in v]
        RVec.append(v)
    return RVec


def Spectral_clustering_labels(L, k, asc_label):
    L = np.array(L)
    L = L.astype(float)
    # calculate smallest k eigenvalues/vectors
    EVal, EVec = sp.sparse.linalg.eigs(L, k=k, which=asc_label)
    # build tarnspose of the eigenvector matrix
    EVec = real_Vec(EVec)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(EVec)
    labels = kmeans.labels_

    return labels


def Spectral_clustering(L, k, asc_label):
    L = np.array(L)
    L = L.astype(float)
    # calculate smallest k eigenvalues/vectors
    EVal, EVec = sp.sparse.linalg.eigs(L, k=k, which=asc_label)
    # build tarnspose of the eigenvector matrix
    EVec = real_Vec(EVec)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(EVec)
    labels = kmeans.labels_
    cluster = []
    for i in range(k):
        index_pos_list = [j for j in range(len(labels)) if labels[j] == i]
        cluster.append(index_pos_list)

    return cluster

# Algorithm 3.1

def SEBA(V, tolerance):
    V = np.array(V)
    p = len(V)
    r = len(V[0])
    mu = 0.99 / np.sqrt(p)
    # mu = 0.06/np.sqrt(p)
    prev_R = np.identity(r, dtype=float)

    V = remove_constant(V, r, p)
    while True:
        S = S_matrix(V, prev_R, r, mu)
        # print(S[0])
        curr_R, H = sp.linalg.polar(S.transpose().dot(V))

        if np.linalg.norm(curr_R - prev_R, 2) <= tolerance:
            break
        else:
            prev_R = curr_R
    for i in range(r):
        S[:, i] = np.sign(sum(S[:, i])) * S[:, i]
    for i in range(r):
        S[:, i] = S[:, i] / max(S[:, i])
    m = [min(S[:, j]) for j in range(r)]
    idx = np.argsort(m)
    idx = idx[::-1]
    S = S[:, idx]
    return S


def remove_constant(V, r, p):
    for j in range(r):
        if max(V[:, j]) - min(V[:, j]) < 1e-14:
            for i in range(p):
                V[i][j] += (np.random.uniform(0, 1) - 0.5) * 1e-12

    return


def C_mu(z, mu):
    return np.sign(z) * max(abs(z) - mu, 0)


def C_mu_Vec(Vec, mu):
    CVec = []
    for v in Vec:
        CVec.append(C_mu(v, mu))
    return CVec


def S_j(V, R, j, mu):
    VR_T = V.dot(R.T)
    C_VR_T = C_mu_Vec(VR_T[:, j], mu)
    return C_VR_T / np.linalg.norm(C_VR_T)


def S_matrix(V, R, r, mu):
    S = []
    for j in range(r):
        S.append(S_j(V, R, j, mu))
    S = np.array(S)
    return S.T


#########################################################

# Algorithm 4.1

def partition_extraction(S):
    S = np.array(S)
    a = []
    p = len(S)
    r = len(S[0])
    for j in range(r):
        S[:, j] = [max(S[i][j], 0) for i in range(p)]

    SS = -np.sort(-S)

    tao = 0
    for i in range(p):
        for j in range(r):
            for l in range(j):
                if sum(SS[i][:l]) > 1 and SS[i][j] > tao:
                    tao = SS[i][j]

    for i in range(p):
        for j in range(r):
            S[i][j] = H_mu(S[i][j], tao)

    for i in range(p):
        j_star = np.argmax(S[i])
        if S[i][j_star] > 0:
            a.append(j_star + 1)
        else:
            a.append(0)
    cluster = []
    for i in range(p + 1):
        index_pos_list = [j for j in range(len(a)) if a[j] == i]
        cluster.append(index_pos_list)

    return cluster

def H_mu(z, mu):
    if abs(z) > mu:
        return z;
    else:
        return 0;
    return np.sign(z) * max(-mu, 0)