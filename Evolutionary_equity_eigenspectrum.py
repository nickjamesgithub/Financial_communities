import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utilities import flatten_dataframe
from sklearn.decomposition import PCA
from scipy.stats.kde import gaussian_kde
from Utilities import coefficient_distribution_plot, flatten_dataframe, dendrogram_plot_test, dendrogram_plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# Import data
# Read in Prices
prices = pd.read_excel("/Users/tassjames/Desktop/Financial_Crises/data/S&P500.xlsx", index_col='Date')
equity_returns = np.log(prices).diff()[1:]
equity_returns.dropna(axis='columns', inplace=True)
equity_returns_length = len(equity_returns)

# PCA
# Correlation Principal components and rolling size of market
corr_value_1 = []
corr_svs = []
eigenspectrum = []

# Smoothing Rate
smoothing_rate = 90

# Smoothing Rate
for i in range(smoothing_rate, equity_returns_length): # len(equity_returns)

    # Returns
    returns = equity_returns.iloc[i-smoothing_rate:i,:]

    # Compute returns correlation with pandas
    correlation = np.nan_to_num(returns.corr())

    # Compute PCA
    pca_corr = PCA(n_components=10)
    pca_corr.fit(correlation)
    corr_value_1.append(np.nan_to_num(pca_corr.explained_variance_ratio_[0]))
    eigenspectrum.append(np.nan_to_num(pca_corr.explained_variance_ratio_))
    print(i)

# Generate date index
date_index_plot = pd.date_range('05-01-2000','08-10-2020',equity_returns_length-smoothing_rate).strftime('%Y-%m-%d')

# Plot explanatory variance exhibited by First eigenvalue
fig, ax = plt.subplots()
plt.plot(date_index_plot, corr_value_1)
plt.title("Explanatory variance of First eigenvalue")
plt.tick_params(axis='x', which='major', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.xlabel("Date")
plt.ylabel("Explanatory variance of eigenvalue 1")
plt.savefig("Eigenvalue_1_time")
plt.show()

# Topology of L^1 distance between explanatory variance of eigenvalues
eigenspectra_topology = np.zeros((len(eigenspectrum), len(eigenspectrum)))
for i in range(len(eigenspectrum)):
    for j in range(len(eigenspectrum)):
        eigen_i = eigenspectrum[i]
        eigen_j = eigenspectrum[j]
        eigenspectra_topology[i,j] = np.sum(np.abs(eigen_i - eigen_j))

eigenspectra_topology_1 = np.zeros((len(corr_value_1), len(corr_value_1)))
for i in range(len(corr_value_1)):
    for j in range(len(corr_value_1)):
        eigen_1_i = corr_value_1[i]
        eigen_1_j = corr_value_1[j]
        eigenspectra_topology_1[i,j] = np.abs(eigen_1_i - eigen_1_j)

# Plot topology of eigenspecta distance
plt.matshow(eigenspectra_topology)
plt.title("Topology of eigenspectra")
plt.savefig("Eigenspectra_topology")
plt.show()

# Plot topology of eigenspecta distance
plt.matshow(eigenspectra_topology_1)
plt.title("Topology of eigenspectra 1")
plt.savefig("Eigenspectra_topology_1")
plt.show()

# Compute distance between eigen 1 and eigen 1-10
consistency = np.abs(eigenspectra_topology - eigenspectra_topology_1)

# Plot topology of eigenspecta distance
plt.matshow(consistency)
plt.title("Difference in topologies")
plt.savefig("Eigenspectra_Consistency")
plt.show()

# Perform Hierarchical clustering on topology of eigenspectra surface
dendrogram_plot(eigenspectra_topology, "_l1_", "_eigenspectra_", date_index_plot)

# Perform Hierarchical clustering on topology of eigenspectra surface
dendrogram_plot(consistency, "_l1_", "_eigen_consistency_", date_index_plot)