import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import wasserstein_distance

# Definition of the Marchenko-Pastur density
def marchenko_pastur_pdf(x, Q, sigma=1):
    y = 1 / Q
    lamda_plus = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    lamda_minus = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    return (1 / (2 * np.pi * sigma**2 * x * y)) * np.sqrt((lamda_plus - x) * (x - lamda_minus)) * (0 if (x > lamda_plus or x < lamda_minus) else 1)

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

    return wasserstein_distance(u_values=e, v_values=x, v_weights=f(x))

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_only.csv", index_col='Date')
prices.dropna(axis='columns', inplace=True)
equity_returns = np.log(prices).diff()[1:]
smoothing_rate_global = 120

# Automatically  rename sector labels
sectors_labels = ["Health Care", "Industrials", "Communication Services", "Information Technology", "Utilities", "Financials",
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

    for j in range(smoothing_rate, len(equity_returns)): # len(equity_returns)
        slice_check = sector_slice.iloc[(j - smoothing_rate):j, :]
        correlation = np.nan_to_num(sector_slice.iloc[(j-smoothing_rate):j, :].corr())
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
    print(sectors_labels[i])
    print(i)

# Set date index for the plot
# Plot all trajectories
total_dist_deviation = []
for i in range(len(sector_trajectories)):
    plt.plot(sector_trajectories[i], label=sectors_labels[i])
    total_dist_deviation.append([sectors_labels[i], np.sum(sector_trajectories[i])])
plt.legend()
plt.savefig("RMT_sector_dist_deviation")
plt.show()

# Make it an array
total_dist_deviation = np.array(total_dist_deviation)
total_dist_dev_ordered = total_dist_deviation[total_dist_deviation[:, 1].argsort()]
print(total_dist_dev_ordered)

