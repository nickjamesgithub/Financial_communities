import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh

# Choose number of sectors and n for simulation
num_sectors = 10
n = 100
sample_per_sector = n//num_sectors

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_sectors.csv", index_col='Date')

# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$','')

sectors_labels = ["Health Care", "Industrials", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy", "Communication Services"]

sectors_labels.sort()

first_eigenvalue_samples = []
portfolio_returns_sample = []
while len(first_eigenvalue_samples) < 2:
    # First pick n sectors at random
    sector_sequence = list(np.linspace(0,len(sectors_labels)-1,len(sectors_labels))) # Randomly draw sector numbers
    random_list_sector = random.sample(sector_sequence, num_sectors)
    ints = [int(item) for item in random_list_sector]

    # Get corresponding sector names
    random_sector_list = []
    for i in range(len(ints)):
        random_sector_drawn = sectors_labels[ints[i]]
        random_sector_list.append(random_sector_drawn)

    # Print random sector list
    print(random_sector_list)

    # Get the random samples for current iteration
    stock_samples = []
    for i in range(len(random_sector_list)):
        sector_slice = prices[random_sector_list[i]]
        length = len(sector_slice.columns)
        random_sequence = list(np.linspace(0, length - 1, length))
        random_list_stocks = random.sample(random_sequence, sample_per_sector)
        ints = [int(item) for item in random_list_stocks]
        random_sector_stocks = sector_slice.iloc[:, ints]
        for j in range(len(random_sector_stocks.iloc[0])):
            stock_slice = random_sector_stocks.iloc[:, j]
            stock_slice_list = list((stock_slice[1:]).astype("float"))
            stock_samples.append(stock_slice_list)

    # Convert back into a dataframe
    stock_samples_df = pd.DataFrame(np.transpose(stock_samples))
    log_returns = np.log(stock_samples_df).diff()[1:]
    smoothing_rate = 120

    returns_list = []
    corr_1 = []
    for i in range(smoothing_rate, len(log_returns)):
        # Returns
        returns = log_returns.iloc[i - smoothing_rate:i, :]
        # Compute with pandas
        correlation = np.nan_to_num(returns.corr())

        # Perform eigendecomposition and get explanatory variance
        m_vals, m_vecs = eigsh(correlation, k=6, which='LM')
        m_vecs = m_vecs[:, -1]  # Get 1st eigenvector
        m_vals_1 = m_vals[-1] / len(correlation)
        corr_1.append(m_vals_1)
        print("Iteration "+str(i)+" / "+str(len(log_returns)))

        # Compute total returns
        returns_1 = np.array(log_returns.iloc[i, :])
        weights = np.repeat(1/len(returns_1), n)
        total_return_iteration = np.sum(returns_1 * weights)
        returns_list.append(total_return_iteration)

    # Append draws of first eigenvalue samples to main list
    first_eigenvalue_samples.append(corr_1)
    portfolio_returns_sample.append(returns_list)

# Plot first 5 samples
avg_explanatory_variance = [] # Average explanatory variance from first eigenvalue
for i in range(len(first_eigenvalue_samples)):
    plt.plot(first_eigenvalue_samples[i])
    avg_explanatory_variance.append(np.mean(first_eigenvalue_samples[i]))
plt.title("First eigenvalue samples")
plt.show()

# Plot distribution of first eigenvalue
plt.hist(avg_explanatory_variance)
plt.title("First eigenvalue average contribution")
plt.show()

# Plot 5 return paths
for i in range(len(portfolio_returns_sample)):
    plt.plot(portfolio_returns_sample[i], alpha=0.1)
plt.title("Portfolio return samples")
plt.show()

# Compute Expected Return
portfolio_returns_sample_array = np.array(portfolio_returns_sample)
returns_expectation = portfolio_returns_sample_array.mean(axis=0)
volatility_expectation = np.std(returns_expectation) * np.sqrt(250)
print(returns_expectation)
print(volatility_expectation)
