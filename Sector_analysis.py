import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_sectors.csv", index_col='Date')

# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$','')

# Sort the sectors
sectors_labels = ["Materials", "Health Care", "Industrials", "Information Technology", "Utilities", "Financials",
            "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy", "Communication Services"]
sectors_labels.sort()

# Model parameters
smoothing_rate = 120
corr_1 = [] # Explanatory variance of first eigenvalue list
inner_product_ones = [] # inner Product ones
inner_product_market = [] # Projection inner product onto market

for s in range(len(sectors_labels)):
    # MARKET
    market = (prices.iloc[2:, :]).astype("float") # Get market prices
    market_returns = np.log(market).diff() # Compute log returns of market

    # SECTOR
    sector_slice = (prices[sectors_labels[s]].iloc[2:,:]).astype("float") # Slice for the sector
    sector_returns = np.log(sector_slice).diff() # Compute log returns of specific sector

    # Store values
    market_ones = []
    corr_1_s = []  # Explanatory variance of first eigenvalue list
    inner_product_ones_s = []  # inner Product ones
    inner_product_market_s = []  # Projection inner product

    for i in range(smoothing_rate, len(sector_returns)): # len(sector_returns)
        # Slice sector returns
        sector_returns_slice = sector_returns.iloc[(i-smoothing_rate):i,:]
        sector_correlation = np.nan_to_num(sector_returns_slice.corr()) # Compute correlation matrix

        # Generate normalized vector of ones
        one_vector = np.ones(len(sector_correlation)) / np.sqrt(len(sector_correlation))  # Vector of ones

        # Eigendecomposition and normalized inner product with unit vector for sector
        s_vals, s_vecs = eigsh(sector_correlation, k=6, which='LM')
        s_vals_1 = s_vals[-1]
        s_vecs_1 = s_vecs[:,-1]

        # Explanatory variance of first sector eigenvalue
        exp_var_1_s = s_vals_1/len(sector_correlation)
        corr_1_s.append(exp_var_1_s)  # Explanatory value first eigenvalue

        # Norm Inner product between 1st eigenvector of market coeffs corresponding to sector and first eigenvector of sector
        norm_inner_product_ones = np.dot(s_vecs_1, one_vector) # Inner product
        inner_product_ones_s.append(norm_inner_product_ones) # Norm Inner product between (111...1) and first eigenvector of sector

        # Take market correlation matrix and perform eigendecomposition
        market_returns_slice = market_returns.iloc[(smoothing_rate - smoothing_rate):smoothing_rate, :]
        market_correlation = np.nan_to_num(market_returns_slice.corr()) # Compute correlation matrix
        m_vals, m_vecs = eigsh(market_correlation, k=6, which='LM')
        m_vals_1 = m_vals[-1] / len(market_correlation) # Get 1st eigenvector
        m_vecs_1 = m_vecs[:,-1] # Get 1st eigenvector

        # Get coefficients of market eigenvalue 1 corresponding to sector
        sector_indices = market.columns.get_loc(sectors_labels[s])
        eig1_sector_coeffs = m_vecs_1[sector_indices]

        # Norm Inner product between 1st eigenvector of market coeffs corresponding to sector and first eigenvector of sector
        norm_inner_product_market_projection = np.dot(eig1_sector_coeffs, s_vecs_1)
        inner_product_market_s.append(norm_inner_product_market_projection)

        # Iteration
        print(sectors_labels[s]+" Simulation "+str(i))

    # Append sector values to overall list
    corr_1.append(corr_1_s)
    inner_product_ones.append(inner_product_ones_s)
    inner_product_market.append(inner_product_market_s)

# Loop over normalized first eigenvalue
for i in range(len(corr_1)):
    plt.plot(corr_1[i], label=sectors_labels[i], alpha=0.45)
plt.ylim(0,1)
plt.title("Explanatory Variance Eigenvalue 1")
plt.legend()
plt.savefig("G_Explanatory_variance_eigenvalue_1")
plt.show()

# Loop over normalized inner product with ones
for i in range(len(inner_product_ones)):
    plt.plot(np.abs(inner_product_ones[i]), label=sectors_labels[i], alpha=0.45)
plt.ylim(0,1)
plt.title("Inner product one_vector")
plt.legend()
plt.savefig("G_inner_product_ones")
plt.show()

# Loop over normalized inner product with sector coeffcs of market first eigenvector
for i in range(len(inner_product_market)):
    plt.plot(np.abs(inner_product_market[i]), label=sectors_labels[i], alpha=0.45)
plt.ylim(0,1)
plt.title("Market projection")
plt.legend()
plt.savefig("G_inner_product_market")
plt.show()






