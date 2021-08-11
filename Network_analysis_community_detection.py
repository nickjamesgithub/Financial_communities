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
from graph_tool.all import *
#
# # plotting setting
# SMALL_SIZE = 12
# MEDIUM_SIZE = 15
# BIGGER_SIZE = 20
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Read in Prices
sp500 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean.csv", index_col='Date')

# Generate log returns
logreturns = np.log(sp500).diff()[1:]

# Generate correlation
correlation = logreturns.corr()

# Plot elements of correlation matrix
entries = np.array(correlation).flatten()

# Histogram of correlation entries
plt.hist(entries, bins=50)
plt.show()

# Generate Graph
G = nx.from_numpy_matrix(np.array(correlation))

def threshold_graph(graph, threshold_score):
    # Remove edges with a small weight
    graph_thresh = graph.copy()
    # delete those edges with a combined score of <= threshold_score (small confidence)
    for edge in graph_thresh.edges:
        weight = list(graph_thresh.get_edge_data(edge[0], edge[1]).values())
        if (weight[0] < threshold_score):
            graph_thresh.remove_edge(edge[0], edge[1])
    return graph_thresh

# Implement Graph Laplacian and Random walk (almost invariant sets)
def Graph_Laplacian_Eigenspectrum(Graph, name):
    A = nx.adjacency_matrix(Graph).todense()
    Lap = nx.laplacian_matrix(Graph).todense() # Graph Laplacian
    Id = np.identity(len(A)) # Identity matrix

    D = Lap + A
    Lr = np.linalg.inv(D).dot(A)
    LR = Id - np.linalg.inv(D).dot(A)

    # Set matrix to be used for spectral clustering
    S = Lap
    s = S.astype(float)

    # Eigendecomposition on Total Matrix
    EVal, EVec = sp.sparse.linalg.eigs(s, k=100, which='SM')

    # Plot graph laplacian
    fig = plt.figure(figsize=(15.0,10.0))
    plt.plot(EVal, marker='o', markersize=10)
    plt.xlabel("i", fontsize=14)
    plt.ylabel("lambda", fontsize=14)
    plt.legend(fontsize='large')
    plt.title(G, fontsize=16)
    plt.savefig("Graph_Laplacian_"+name)
    plt.show()

# Threshold Graph over Full time period
graph_thresh = threshold_graph(G, 0.3)

# Plot Eigenspectrum of Graph Laplacian for full graph
# and threshold graph over full time period
Graph_Laplacian_Eigenspectrum(G, "G")
Graph_Laplacian_Eigenspectrum(graph_thresh, "G_threshold")

# # Market periods
# # Slice different market periods
# gfc = logreturns.iloc[1827:2717,:]
# gfc_crash = logreturns.iloc[2263:2457,:]
# interim = logreturns.iloc[2718:5261,:]
# covid = logreturns.iloc[5262:5392,:]
# covid_crash = logreturns.iloc[5262:5326,:]
#
# # Compute with pandas
# gfc_correlation = np.nan_to_num(np.array(gfc.corr()))
# gfc_crash_correlation = np.nan_to_num(np.array(gfc_crash.corr())) #+ 0.00001 * np.identity(len(gfc_crash.corr()))
# interim_correlation = np.nan_to_num(np.array(interim.corr()))
# covid_correlation = np.nan_to_num(np.array(covid.corr()))
# covid_crash_correlation = np.nan_to_num(np.array(covid_crash.corr()))
#
# # Generate graphs for different periods in the market
# G_gfc = nx.from_numpy_matrix(gfc_correlation)
# G_gfc_crash = nx.from_numpy_matrix(gfc_crash_correlation)
# G_interim = nx.from_numpy_matrix(interim_correlation)
# G_covid = nx.from_numpy_matrix(covid_correlation)
# G_covid_crash = nx.from_numpy_matrix(covid_crash_correlation)
#
# # Implement Graph Laplacian and Random walk (almost invariant sets)
# def graph_laplacian(graph, name):
#     g = graph
#     A = nx.adjacency_matrix(g).todense()
#     Lap = nx.laplacian_matrix(g).todense()
#     Id = np.identity(len(A))
#
#     D = Lap + A
#     Lr = np.linalg.inv(D).dot(A)
#     LR = Id - np.linalg.inv(D).dot(A)
#
#     # Set matrix to be used for spectral clustering
#     S = Lap
#     s = S.astype(float)
#     EVal, EVec = sp.sparse.linalg.eigs(s, k=50, which='SR')
#
#     # Plot graph laplacian
#     fig = plt.figure(figsize=(15.0,10.0))
#     plt.plot(EVal, marker='o', markersize=10)
#     plt.xlabel("i", fontsize=14)
#     plt.ylabel("lambda", fontsize=14)
#     plt.legend(fontsize='large')
#     plt.title(name, fontsize=16)
#     plt.savefig("Graph_Laplacian_"+name)
#     plt.show()
#
# # Plot Graph laplacian
# graph_laplacian(G_gfc, "GFC_")
# graph_laplacian(G_gfc_crash, "GFC_Crash_")
# graph_laplacian(G_interim, "Interim_")
# graph_laplacian(G_covid, "COVID_")
# graph_laplacian(G_covid_crash, "COVID_crash_")
