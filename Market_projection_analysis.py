import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_sectors.csv", index_col='Date')
market = (prices.iloc[2:, :]).astype("float") # Get market prices
market_returns = np.log(market).diff() # Compute log returns of market

# Model parameters
smoothing_rate = 120
corr_1 = [] # Explanatory variance of first eigenvalue list
inner_product_ones = [] # inner Product ones

for i in range(smoothing_rate, len(market_returns)): # len(market_returns)

    # Take market correlation matrix
    market_returns_slice = market_returns.iloc[(i - smoothing_rate):i, :]
    market_correlation = np.nan_to_num(market_returns_slice.corr())  # Compute correlation matrix
    m_vals, m_vecs = eigsh(market_correlation, k=6, which='LM')
    m_vecs = m_vecs[:, -1]  # Get 1st eigenvector
    m_vals_1 = m_vals[-1]/len(market_correlation)
    corr_1.append(m_vals_1)

    # Inner product with ones
    one_vector = np.ones(len(market_correlation))/np.sqrt(len(market_correlation))
    norm_inner_product_ones = np.dot(m_vecs, one_vector)
    norm_inner_product = np.dot(m_vecs, np.ones(len(market_correlation)))\
                            /(np.linalg.norm(m_vecs) * np.linalg.norm(np.ones(len(market_correlation))))
    norm_1 = np.linalg.norm(m_vecs)
    norm_2 = np.linalg.norm(one_vector)
    # norm_inner_product_ones = np.abs(np.dot(s_vecs, one_vector)/(np.linalg.norm(s_vecs) * np.linalg.norm(one_vector))) # Inner product
    inner_product_ones.append(norm_inner_product) # Norm Inner product between (111...1) and first eigenvector of sector

    print(" Simulation " + str(i))

# Normalized eigenvalue 1
plt.plot(corr_1)
plt.title("Market normalized eigenvalue 1")
plt.savefig("G_Market_1")
plt.show()

# Inner product ones
plt.plot(np.abs(inner_product_ones))
plt.title("Market and Ones IP")
plt.savefig("G_Market_IP")
plt.show()






