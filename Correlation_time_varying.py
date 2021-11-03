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

# Correlation paths
correlation_time_varying = []
for s in range(len(sectors_labels)):
    # MARKET
    market = (prices.iloc[2:, :]).astype("float") # Get market prices
    market_returns = np.log(market).diff() # Compute log returns of market

    # SECTOR
    sector_slice = (prices[sectors_labels[s]].iloc[2:,:]).astype("float") # Slice for the sector
    sector_returns = np.log(sector_slice).diff() # Compute log returns of specific sector

    correlation_path = [] # Correlation path
    for i in range(smoothing_rate, len(sector_returns)): # len(sector_returns)
        # Slice sector returns
        sector_returns_slice = sector_returns.iloc[(i-smoothing_rate):i,:]
        sector_correlation = np.nan_to_num(sector_returns_slice.corr()) # Compute correlation matrix
        # upper_triangle = np.triu(sector_correlation, k=0)
        avg_correlation = np.mean(sector_correlation)
        correlation_path.append(avg_correlation) # Append correlation path
        print("Sector " + str(s) + " Iteration " + str(i))
    # Append time varying correlation path
    correlation_time_varying.append(correlation_path)

# Plot all time varying paths
# Generate date index
date_index_plot = pd.date_range('16-06-2000','08-10-2020',len(correlation_time_varying[0])).strftime('%Y-%m-%d')

fig,ax = plt.subplots()
for i in range(len(correlation_time_varying)):
    plt.plot(date_index_plot, correlation_time_varying[i], label=sectors_labels[i], alpha=0.35)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.tick_params(axis='x', which='major', labelsize=8)
plt.legend()
plt.savefig("Time_varying_correlation")
plt.show()