import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import wasserstein_distance

# Definition of the Marchenko-Pastur density
def marchenko_pastur_pdf(x, Q, sigma=1):
    y = 1 / Q
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    return (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a)) * (0 if (x > b or x < a) else 1)

def eigenvalue_distribution_distance(correlation_matrix, Q, sigma=1, bins=30):
    # Eigendecomposition for Hermitian matrix
    e, _ = np.linalg.eig(correlation_matrix)  # Correlation matrix is Hermitian, so this is faster
    # than other variants of eig

    # Set the Grid
    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    # Plot the theoretical density
    f = np.vectorize(lambda x: marchenko_pastur_pdf(x, Q, sigma=sigma))
    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)
    x = np.linspace(x_min, x_max, 5000)

    return wasserstein_distance(e, f(x))

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_only.csv", index_col='Date')
prices.dropna(axis='columns', inplace=True)
equity_returns = np.log(prices).diff()[1:]
smoothing_rate_global = 180

# Automatically  rename sector labels
sectors_labels = ["Industrials", "Health Care", "Communication Services", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary"]

# Replace column names for prices and returns
prices.columns = prices.columns.str.replace('(\.\d+)$','')
equity_returns.columns = equity_returns.columns.str.replace('(\.\d+)$','')

# Loop over sectors and generate rolling correlation matrix a
sector_trajectories = []
for i in range(len(sectors_labels)): # len(sector_labels)
    sector_slice = equity_returns[sectors_labels[i]]
    theoretical_empirical_dist = []

    # Smoothing rate
    smoothing_rate = smoothing_rate_global

    for j in range(smoothing_rate, 1000): # len(equity_returns)
        correlation = sector_slice.iloc[(j-smoothing_rate):j, :].corr()
        T = smoothing_rate_global
        N = sector_slice.shape[1]  # Pandas does the reverse of what I wrote in the first section
        Q = T / N

        # Compute distance between theoretical and empirical distribution
        dist = eigenvalue_distribution_distance(correlation, Q, sigma=1)

        # Append to list
        theoretical_empirical_dist.append(dist)

        print(j)

    # Append full trajectory to sector trajectories
    sector_trajectories.append(theoretical_empirical_dist)
    print(i)

# Set date index for the plot
# date_index = pd.date_range('01-01-2000','08-10-2020',len(equity_returns)).strftime('%Y-%m-%d')
# date_index_slice = date_index[smoothing_rate_global:]
# Plot all trajectories
for i in range(len(sector_trajectories)):
    plt.plot(sector_trajectories[i], label=sectors_labels[i])
plt.legend()
plt.show()

