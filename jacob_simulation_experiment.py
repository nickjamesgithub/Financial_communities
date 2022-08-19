import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh

# Choose number of sectors and n for simulation
num_simulations = 1  # 500
global_paths = []
# Store portfolio grid values in sample

while len(global_paths) < num_simulations:
    portfolio_grid_samples = []
    sectors_list = [2,3,4,5] # 2,3,4,5,6,7,8,9
    samples_list = [2,3,4,5] # 2,3,4,5,6,7,8,9
    for k in range(len(sectors_list)):
        for s in range(len(samples_list)):

            num_sectors = sectors_list[k]
            sample_per_sector = samples_list[s]

            # n is samples per sector * number of sectors
            n = sample_per_sector * num_sectors

            # Import data
            prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors_clean.csv", index_col='Date')

            # Replace column names for prices
            prices = prices.reindex(sorted(prices.columns), axis=1)
            prices.columns = prices.columns.str.replace('(\.\d+)$','')

            sectors_labels = ["Health Care", "Industrials", "Information Technology", "Utilities", "Financials",
                       "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy", "Communication Services"]

            sectors_labels.sort()

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
            smoothing_rate = 4500 # set to 120

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
                print("Sectors ", sectors_list[k], " Samples", samples_list[s])
                print("Iteration "+str(i)+" / "+str(len(log_returns)))

                # Compute total returns
                returns_1 = np.array(log_returns.iloc[i, :])
                weights = np.repeat(1/len(returns_1), n)
                total_return_iteration = np.sum(returns_1 * weights)
                returns_list.append(total_return_iteration)

            # Compute average lambda_1(t) value over the time period
            avg_corr_1 = np.mean(corr_1)
            portfolio_grid_samples.append(avg_corr_1)

    # Make portfolio grid an array
    portfolio_grid_samples_array = np.array(portfolio_grid_samples)
    portfolio_grid_reshaped = portfolio_grid_samples_array.reshape(len(sectors_list), len(samples_list))

    def matrix_path_function(array):
        steps_list = []
        down_counter = 0
        right_counter = 0
        while len(steps_list) < 2 * len(array):
            if down_counter + 1 >= len(array) or right_counter + 1 >= len(array):
                print(steps_list)
                return steps_list
            else:
                if array[down_counter,right_counter+1] < array[down_counter+1,right_counter]:
                    move = "right"
                    steps_list.append(move)
                    right_counter += 1
                else:
                    move = "down"
                    steps_list.append(move)
                    down_counter += 1

    path = matrix_path_function(portfolio_grid_reshaped)
    block = 1