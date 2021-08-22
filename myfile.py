import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(1234)

# Choose number of sectors and n for simulation
num_sectors = 4
n = 30
sample_per_sector = n//num_sectors

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_sectors.csv", index_col='Date')

# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$','')

sectors_labels = ["Health Care", "Industrials", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy", "Communication Services"]
# sectors_labels = ["Consumer Discretionary", "Energy", "Communication Services", "Energy"]

sectors_labels.sort()

first_eigenvalue_samples = []

while len(first_eigenvalue_samples) < 5:
    # First pick n sectors at random
    # sector_sequence = list(np.linspace(0,9,10))
    sector_sequence = list(np.linspace(0,10,11)) # Randomly draw sector numbers
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

#     for i in range(len(random_sector_list)):
#         sector_slice = prices[random_sector_list[i]]
#         print(random_sector_list[i])
#         length = len(sector_slice.columns)
#         random_sequence = list(np.linspace(0, length-1, length))
#         random_list = random.sample(sector_sequence, sample_per_sector)
#         ints = [int(item) for item in random_list]
#         random_sector_stocks = sector_slice.iloc[:,ints]
#         for j in range(len(random_sector_stocks.iloc[0])):
#             stock_slice = random_sector_stocks.iloc[:,j]
#             stock_slice_list = list((stock_slice[1:]).astype("float"))
#             stock_samples.append(stock_slice_list)

    # Convert back into a dataframe
    stock_samples_df = pd.DataFrame(np.transpose(stock_samples))
    log_returns = stock_samples_df.diff()[1:]
    smoothing_rate = 120

    corr_1 = []
    for i in range(smoothing_rate, len(log_returns)):
        # Returns
        returns = log_returns.iloc[i - smoothing_rate:i, :]
        # Compute with pandas
        correlation = np.nan_to_num(returns.corr())

        # Compute PCA
        pca_corr = PCA(n_components=10)
        pca_corr.fit(correlation)
        corr_1.append(pca_corr.explained_variance_ratio_[0])
        print("Iteration "+str(i)+" / "+str(len(log_returns)))

    # Append draws of first eigenvalue samples to main list
    first_eigenvalue_samples.append(corr_1)

# Plot first 5 samples
for i in range(len(first_eigenvalue_samples)):
    plt.plot(first_eigenvalue_samples[i])
plt.title("First eigenvalue samples")
plt.show()