import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import wasserstein_distance
from scipy.sparse import linalg

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_only.csv", index_col='Date')
prices.dropna(axis='columns', inplace=True)
equity_returns = np.log(prices).diff()[1:]

# Automatically  rename sector labels
sectors_labels = ["#N/A", "Health Care", "Industrials", "Communication Services", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy"]
sectors_labels.sort()

# Replace column names for prices and returns
prices = prices.reindex(sorted(prices.columns), axis=1)
prices_df = pd.DataFrame(prices)
prices_df.to_csv("/Users/tassjames/Desktop/prices_ordered.csv")
equity_returns = equity_returns.reindex(sorted(equity_returns.columns), axis=1)

# Eigendecomposition
vals, vecs = linalg.eigsh(np.nan_to_num(equity_returns.corr()))
vecs_array = np.array(vecs)
vecs_array_slice = vecs_array[:,1:5]

# Construct Normalised eigenvectors test
normalized_eigenvectors = []
for i in range(len(vecs_array_slice[0])):
    eigenvector_raw = vecs_array_slice[:,i]
    normalisation = np.linalg.norm(vecs_array_slice[:,i])
    norm_eigenvector = eigenvector_raw/normalisation
    normalized_eigenvectors.append(norm_eigenvector)

# Get number of stocks in each sector
sectors_lengths = [] #
for i in range(len(sectors_labels)):
    sector_names = equity_returns.filter(like=sectors_labels[i])
    length = len(sector_names.iloc[0])
    sectors_lengths.append(length) #

# Plot normalized eigenvectors
eigen_labels = [2,3,4,5]
cps = np.cumsum(sectors_lengths)
grid = np.linspace(0, len(normalized_eigenvectors[0]), len(normalized_eigenvectors[0]))
for i in range(len(normalized_eigenvectors)):
    plt.bar(grid, np.abs(normalized_eigenvectors[i])**2)
    for j in range(len(cps)):
        plt.axvline(cps[j], color='red', alpha=0.25)
    plt.title("Eigenvector "+str(eigen_labels[i]))
    plt.show()
