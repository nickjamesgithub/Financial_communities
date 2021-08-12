import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import numpy.linalg as la
import scipy as sp
import pandas as pd
import csv
import pickle
import xlrd
# Ignore warnings
import openpyxl
import warnings
warnings.filterwarnings('ignore')

# networkx
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import k_clique_communities
# from community import community_louvain
import community_louvain
import scipy.sparse.linalg
import sklearn
from sklearn.cluster import KMeans
import sklearn.preprocessing
from sklearn.decomposition import PCA
import itertools
from itertools import chain
from Network_helper_functions import Spectral_clustering

# graph-tools
sys.path.append('/usr/local/Cellar/graph-tool/2.33/lib/python3.8/site-packages/')
from graph_tool.all import *

# plotting setting
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Read in Prices
sp500 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean.csv", index_col='Date')
N = len(sp500)
print('N:',N)

# Generate log returns
logreturns = np.log(sp500).diff()[1:]
print(logreturns)

# Compute with pandas
correlation = logreturns.corr()

# Entries of correlation matrix
weights = np.array(correlation).flatten()
i=0
w = []
for ww in weights:
    i = i+1
    w.append(ww)
w.sort(reverse=True)
type(w)

# Plot elements of correlation matrix
fig = plt.figure(figsize=(15.0,10.0))
plt.plot(w, marker='o', markersize=10)
plt.xlabel("i")
plt.ylabel("weights")
plt.show()

# Generate adjacency matrix from correlation matrix
correlation = np.array(correlation) # Make correlation matrix an array
correlation_network = np.absolute(correlation)
correlation_network = correlation_network - correlation_network.diagonal()
G0 = nx.from_numpy_matrix(correlation)

print('number of nodes of the full G:',len(G0.nodes))
print('Is the full G connected?',nx.connected.is_connected(G0))
print('How many connected subgraphs are there?',nx.connected.number_connected_components(G0))

# Draw the network
nx.draw(G0, node_size=15, alpha=0.4, node_color="blue", with_labels=True)
plt.show()

layout_PPI = nx.spring_layout(G0,k=1/np.sqrt(len(G0)))
plt.rcParams.update({'figure.figsize': (15, 10)})
nx.draw_networkx(G0, pos=layout_PPI, node_size=0, edge_color="#444444", alpha=0.05, with_labels=False)

# Weights
print(G0[34][346]["weight"])
set(chain.from_iterable(d.keys() for *_, d in G0.edges(data=True)))
# {'length', 'weight'}

for node in G0.nodes():
    edges = G0.edges(node, data=True)
    if len(edges) > 1:  # some nodes have zero edges going into it
        max_weight = max([edge[2]['weight'] for edge in edges])
        min_weight = min([edge[2]['weight'] for edge in edges])
        g0_weights = [edge[2]['weight'] for edge in edges]

print('G0 maximal weight:', max_weight)
print('G0 minimal weight:', min_weight)
g0_weights.sort(reverse=True)

# # Plot G0
# fig = plt.figure(figsize=(15.0, 10.0))
# plt.plot(g0_weights, marker='o', markersize=10)
# plt.xlabel("i")
# plt.ylabel("weights")
# plt.show()

# Set threshold score
threshold_score = 0.2
# Remove edges with a small weight
G1 = G0
for node in G1.nodes():
    edges = G1.edges(node, data=True)
    if len(edges) > 1:  # some nodes may have zero edges going into it
        max_weight = max([edge[2]['weight'] for edge in edges])
        # idx = np.where((k - delta <= X) * (X <= k))[0].max()
        # print(max_weight)
        for edge in list(edges):
            if edge[2]['weight'] < threshold_score: # was max_weight
                """CHECK THIS"""
                G1.remove_edge(edge[0], edge[1])

# Weights
# print(G1[34][346]["weight"])
set(chain.from_iterable(d.keys() for *_, d in G1.edges(data=True)))
# {'length', 'weight'}

for node in G1.nodes():
    edges = G1.edges(node, data=True)
    if len(edges) > 1:  # some nodes have zero edges going into it
        max_weight = max([edge[2]['weight'] for edge in edges])
        min_weight = min([edge[2]['weight'] for edge in edges])
        g1_weights = [edge[2]['weight'] for edge in edges]

print('G1 maximal weight:', max_weight)
print('G1 minimal weight:', min_weight)
g1_weights.sort(reverse=True)

# Plot G1
fig = plt.figure(figsize=(15.0, 10.0))
plt.plot(g0_weights, marker='o', markersize=10, label="g0 weights")
plt.plot(g1_weights, marker='o', markersize=10, label="g1 weights")
plt.xlabel("i")
plt.ylabel("weights")
plt.legend()
plt.show()

# Plot G1, smaller network
layout_PPI = nx.spring_layout(G1, k=1 / np.sqrt(len(G0)))
plt.rcParams.update({'figure.figsize': (15, 10)})
nx.draw_networkx(G1, pos=layout_PPI, node_size=0, edge_color="#444444", alpha=0.05, with_labels=False)
plt.show()

# Degree list
dgList=[]
# print('asd',G0.degree)
for i in G0.degree():
    dgList.append([i[1],i[0]])
dgList.sort()
dgList.reverse()
topN = 10
print(dgList[:topN])
for j in dgList[:topN]:
    print("Node "+str(j[1])+" has degree "+str(j[0]))

print('number of nodes of the full G:',len(G0.nodes))
print('Is the full G connected?',nx.connected.is_connected(G0))
print('How many connected subgraphs are there?',nx.connected.number_connected_components(G0))

# delete those edges with a combined score of <= threshold_score (small confidence)
threshold_score = 0.1
#threshold_score = 0

for edge in G0.edges:
    G0.get_edge_data(edge[0],edge[1])
    weight = list(G0.get_edge_data(edge[0],edge[1]).values())
    #print('qwe',weight[0])
    if(weight[0] <= threshold_score):
        G0.remove_edge(edge[0],edge[1])

# restrict to largest connected component
largest_cc = max(nx.connected_components(G0),key=len)
G0 = G0.subgraph(largest_cc)
print('number of nodes of thresholded full G:',len(G0.nodes))

#Gc = max(nx.connected_component_subgraphs(G0), key=len)
print('number of nodes of largest connected component of G:',len(G0.nodes))
print('number of nodes of G0:',nx.number_of_nodes(G0))
print('number of edges of G0:',nx.number_of_edges(G0))

"Generate a reduced network which only contains nodes with degrees " \
" smaller than some threshold"

# degree list
dgList=[]
for i in G0.degree():
    dgList.append([i[1],i[0]])
dgList.sort()
dgList.reverse()

deg_cut = 200
highDeg = [dgList[i][1] for i in range(deg_cut)]
print('Largest degree is:',G0.degree(highDeg[0]))
print('300th largest degree is:',G0.degree(highDeg[deg_cut-1]))

# subgraph centrality list (Estrada and Rodriguez-Velazquez, PRE 2005)
sgcList=[]
for i in nx.subgraph_centrality(G0).items():
    sgcList.append([i[1],i[0]])
sgcList.sort()
sgcList.reverse()

sgc_cut = 200
highSGC = [sgcList[i][1] for i in range(sgc_cut)]

# prune high degree nodes
Gr = G0.copy()
Gr.remove_nodes_from(highDeg)
print('Is the pruned Gr connected?', nx.connected.is_connected(Gr))
print('Number of connected components of the pruned network Gr', nx.number_connected_components(Gr))

i = 0
lenCC = []
for comp in nx.connected_components(Gr):
    i = i + 1
    # print('Component %i',i,'has %i nodes',len(comp))
    lenCC.append(len(comp))

lenCC.sort(reverse=True)
for lencc in lenCC[0:10]:
    print('Component length (degree):', lencc)

# restrict to largest connected component if there are lots of isolated nodes
largest_cc = max(nx.connected_components(Gr), key=len)
Gr = Gr.subgraph(largest_cc)
print('Number of nodes of largest connected component of pruned Gr:', len(Gr.nodes))
print('Number of edges of largest connected component of pruned Gr:', len(Gr.edges))

# Prune high SGC nodes
Gs = G0.copy()
Gs.remove_nodes_from(highSGC)
print('Is the pruned Gs connected?', nx.connected.is_connected(Gs))
print('Number of connected components of the pruned network Gs', nx.number_connected_components(Gs))

i = 0
lenCC = []
for comp in nx.connected_components(Gs):
    i = i + 1
    # print('Component %i',i,'has %i nodes',len(comp))
    lenCC.append(len(comp))

lenCC.sort(reverse=True)
for lencc in lenCC[0:10]:
    print('Component length (degree):', lencc)

# restrict to largest connected component if there are lots of isolated nodes
largest_cc = max(nx.connected_components(Gs), key=len)
Gs = Gs.subgraph(largest_cc)
print('Number of nodes of largest connected component of pruned Gs:', len(Gs.nodes))
print('Number of edges of largest connected component of pruned Gs:', len(Gs.edges))

# Implement Graph Laplacian and Random walk (almost invariant sets)

# We use either Gs or Gr
g = Gs
A = nx.adjacency_matrix(g).todense()
Lap = nx.laplacian_matrix(g).todense()
Id =  np.identity(len(A))

D = Lap + A
Lr = np.linalg.inv(D).dot(A)
LR = Id - np.linalg.inv(D).dot(A)

# Set matrix to be used for spectral clustering
S = Lap
s = S.astype(float)
EVal, EVec = sp.sparse.linalg.eigs(s,k=100,which='SR')

# Plot graph laplacian
fig = plt.figure(figsize=(15.0,10.0))
plt.plot(EVal, marker='o', markersize=10)
plt.xlabel("i")
plt.ylabel("lambda")
plt.legend(fontsize='large')
plt.show()

# read off spectral gap from previous plot of spectrum
numb_comm = 9
S = Lap
clusters = Spectral_clustering(S,numb_comm,'SR')

i=0
lenCC = []
for comp in clusters:
    #print('Component %i',i,'has %i nodes',len(comp))
    lenCC.append(len(comp))
    i = i+1

lenCC.sort(reverse=True)
i = 0
for lencc in lenCC[:]:
    print('Community size of #',i,'is:',lencc)
    i = i+1

# Set matrix to be used for spectral clustering
S = Lr

# need largest eigenvalues for Lr
s = S.astype(float)
EVal, EVec = sp.sparse.linalg.eigs(s,k=50,which='SR')
fig = plt.figure(figsize=(15.0,10.0))
plt.plot(EVal, marker='o', markersize=10)
plt.xlabel("i")
plt.ylabel("lambda")
plt.legend(fontsize='large')
plt.show()

