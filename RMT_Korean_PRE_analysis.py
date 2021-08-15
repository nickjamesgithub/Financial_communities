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
sectors_labels = ["Health Care", "Industrials", "Communication Services", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy"]

# Replace column names for prices and returns
equity_returns = equity_returns.reindex(sorted(equity_returns.columns), axis=1)

# Eigendecomposition
vals, vecs = linalg.eigsh(np.nan_to_num(equity_returns.corr()))
vecs_array = np.array(vecs)
vecs_array_slice = vecs_array[:,1:5]

# Construct Normalised eigenvectors
normalized_eigenvectors = []
for i in range(len(vecs_array_slice[0])):
    eigenvector_raw = vecs_array_slice[:,i]
    normalisation = np.sqrt(np.sum(vecs_array_slice[:,i]**2))
    norm_eigenvector = eigenvector_raw/normalisation
    normalized_eigenvectors.append(norm_eigenvector)

# Plot normalized eigenvectors
eigen_labels = [2,3,4,5]
for i in range(len(normalized_eigenvectors)):
    plt.hist(normalized_eigenvectors[i], bins=346)
    plt.title("Eigenvector "+str(eigen_labels[i]))
    plt.show()
