import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigs

# Choose number of sectors and n for simulation
num_sectors = 10
n = 100
sample_per_sector = n//num_sectors

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_sectors.csv", index_col='Date')

# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$','')

# Sort the sectors
sectors_labels = ["Health Care", "Industrials", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy", "Communication Services"]
sectors_labels.sort()

# Model parameters
smoothing_rate = 120
corr_1 = [] # Explanatory variance of first eigenvalue list
inner_product_ones = [] # inner Product ones
projection_inner_product = [] # Projection inner product

for s in range(len(sectors_labels)):
    # MARKET
    market = (prices.iloc[2:, :]).astype("float") # Get market prices
    market_returns = np.log(market).diff() # Compute log returns of market

    # SECTOR
    sector_slice = (prices[sectors_labels[s]].iloc[2:,:]).astype("float") # Slice for the sector
    sector_returns = np.log(sector_slice).diff() # Compute log returns of specific sector

    for i in range(len(sector_returns)):
        sector_returns_slice = sector_returns.iloc[(smoothing_rate-smoothing_rate):smoothing_rate,:]
        sector_correlation = np.nan_to_num(sector_returns_slice.corr()) # Compute correlation matrix

        # Compute PCA and study normalized first eigenvalue
        pca_corr = PCA(n_components=10)
        pca_corr.fit(sector_correlation)
        corr_1.append(pca_corr.explained_variance_ratio_[0]) # First normalized eigenvalue

        # Eigendecomposition and normalized inner product with unit vector for sector
        s_vals, s_vecs = eigs(sector_correlation, k=1)
        s_vecs = np.reshape(s_vecs, (len(s_vecs),1)).flatten()
        one_vector = np.ones(len(sector_correlation))
        norm_inner_product_ones = np.dot(s_vecs, one_vector)/(np.linalg.norm(s_vecs) * np.linalg.norm(one_vector))
        inner_product_ones.append(norm_inner_product_ones) # Norm Inner product between (111...1) and first eigenvector of sector

        # Take market correlation matrix
        market_returns_slice = market_returns.iloc[(smoothing_rate - smoothing_rate):smoothing_rate, :]
        market_correlation = np.nan_to_num(market_returns_slice.corr()) # Compute correlation matrix
        m_vals, m_vecs = eigs(market_correlation, k=1)
        m_vecs = np.reshape(m_vecs, (len(m_vecs), 1)).flatten()

        # Get coefficients of market eigenvalue 1 corresponding to sector
        sector_indices = market.columns.get_loc(sectors_labels[s])
        eig1_sector_coeffs = m_vecs[sector_indices]
        norm_inner_product_market_projection = np.dot(eig1_sector_coeffs, s_vecs)/(np.linalg.norm(s_vecs) * np.linalg.norm(eig1_sector_coeffs))
        projection_inner_product.append(norm_inner_product_market_projection) # Norm Inner product between 1st eigenvector of market coeffs corresponding to sector and first eigenvector of sector

        # Iteration
        print(sectors_labels[s]+" Simulation "+str(i))

# Loop over normalized first eigenvalue
for i in range(len(corr_1)):
    plt.plot(corr_1[i])
    plt.title("Explanatory Variance Eigenvalue 1")
plt.savefig("G_Explanatory_variance_eigenvalue_1")
plt.show()

# Loop over normalized inner product with ones
for i in range(len(inner_product_ones)):
    plt.plot(inner_product_ones[i])
    plt.title("Inner product ones")
plt.savefig("G_inner_product_ones")
plt.show()

# Loop over normalized inner product with sector coeffcs of market first eigenvector
for i in range(len(projection_inner_product)):
    plt.plot(projection_inner_product[i])
    plt.title("Market projection")
plt.savefig("G_inner_product_market")
plt.show()






