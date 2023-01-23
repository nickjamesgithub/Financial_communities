import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors_clean.csv", index_col='Date')
# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$', '')

# Get unique column names for sectors
unique_columns = prices.columns.unique()

# Write out all the slices for various economic periods
dot_com = prices.iloc[146:794, :]
gfc = prices.iloc[1865:2703, :]
covid = prices.iloc[5184:5304, :]
ukraine = prices.iloc[5642:5733, :]

# Log returns of all economic periods
dot_com_returns = np.log(dot_com).diff()[1:]
gfc_returns = np.log(gfc).diff()[1:]
covid_returns = np.log(covid).diff()[1:]
ukraine_returns = np.log(ukraine).diff()[1:]
x=1
y=2


# Correlation matrix for each period
dot_com_correlation = dot_com_returns.corr()
gfc_correlation = gfc_returns.corr()
covid_correlation = covid_returns.corr()
ukraine_correlation = ukraine_returns.corr()

# Plot heatmaps
# Dotcom
plt.matshow(dot_com_correlation)
plt.title("Dot com")
plt.colorbar()
plt.show()

# GFC
plt.matshow(gfc_correlation)
plt.title("GFC")
plt.colorbar()
plt.show()

# covid-19
plt.matshow(covid_correlation)
plt.title("COVID-19")
plt.colorbar()
plt.show()

# Ukraine
plt.matshow(ukraine_correlation)
plt.title("Ukraine Crisis")
plt.colorbar()
plt.show()



