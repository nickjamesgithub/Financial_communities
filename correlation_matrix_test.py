import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_sectors.csv", index_col='Date')

# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$','')

# Convert back into a dataframe
prices_slice = prices.iloc[2:,:]
log_returns = np.log(prices.iloc[2:,:]).diff()[1:]

# Returns
smoothing_rate = 150
returns = log_returns.iloc[150 - smoothing_rate:150, :]
# Compute with pandas
correlation = np.nan_to_num(returns.corr())
corr_diag = correlation - np.identity(len(correlation))

plt.matshow(corr_diag)
plt.colorbar()
plt.show()