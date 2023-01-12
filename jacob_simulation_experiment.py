import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh

crisis_list = ["dot_com", "gfc", "covid", "ukraine"]
# crisis_list = ["dot_com"]

for c in range(len(crisis_list)):
    crisis = crisis_list[c] # dot_com, gfc, covid, ukraine

    # Import data
    prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors_clean.csv", index_col='Date')

    if crisis == "dot_com":
        prices = prices.iloc[146:794,:]

    if crisis == "gfc":
        prices = prices.iloc[1865:2703,:]

    if crisis == "covid":
        prices = prices.iloc[5184:5304,:]

    if crisis == "ukraine":
        prices = prices.iloc[5642:5733,:]

    # Choose number of sectors and n for simulation
    num_simulations = 50
    grid_runs = 5
    global_paths = []
    xs_path = []
    ys_path = []

    # Store portfolio grid values in sample
    while len(global_paths) < num_simulations:
        portfolio_grid_samples = []
        sectors_list = [2,3,4,5,6,7,8,9] # 2,3,4,5,6
        samples_list = [2,3,4,5,6,7,8,9] # 2,3,4,5,6,7,8,9
        for k in range(len(sectors_list)):
            for s in range(len(samples_list)):

                num_sectors = sectors_list[k]
                sample_per_sector = samples_list[s]

                # n is samples per sector * number of sectors
                n = sample_per_sector * num_sectors

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
                smoothing_rate = 90 # set to 90

                returns_list = []
                corr_global = []
                while len(corr_global) < grid_runs:
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
                        print("Correlation run length", len(corr_global) + 1)

                    # Append first eigenvalue path to global run
                    corr_global.append(np.mean(corr_1))

                # Compute average lambda_1(t) value over the time period
                avg_corr_1_global = np.mean(corr_global)
                portfolio_grid_samples.append(avg_corr_1_global)

        # Make portfolio grid an array
        portfolio_grid_samples_array = np.array(portfolio_grid_samples)
        portfolio_grid_reshaped = portfolio_grid_samples_array.reshape(len(sectors_list), len(samples_list))

        def matrix_path_function(array):
            steps_list = []
            x_list = []
            y_list = []
            down_counter = 0
            right_counter = 0
            x_counter = 0
            y_counter = 0
            while len(steps_list) < 2 * len(array):
                if down_counter + 1 >= len(array) or right_counter + 1 >= len(array):
                    print(steps_list)
                    return x_list, y_list, steps_list
                else:
                    if array[down_counter,right_counter+1] < array[down_counter+1,right_counter]:
                        move = "right"
                        steps_list.append(move)
                        x_list.append(x_counter)
                        right_counter += 1
                        x_counter += 1
                        y_list.append(y_counter)
                    else:
                        move = "down"
                        steps_list.append(move)
                        x_list.append(x_counter)
                        down_counter += 1
                        x_counter += 1
                        y_counter -= 1
                        y_list.append(y_counter)

        # Append xs, ys, full path
        xs, ys, path = matrix_path_function(portfolio_grid_reshaped)
        xs_path.append(xs)
        ys_path.append(ys)
        global_paths.append(path)

    # Convert all to dataframes
    global_paths_df = pd.DataFrame(global_paths)
    xs_df = pd.DataFrame(xs_path)
    ys_df = pd.DataFrame(ys_path)

    # Write to csv file
    global_paths_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/global_paths_jacob_"+crisis+".csv")
    xs_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/xs_df_"+crisis+".csv")
    ys_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/ys_df_"+crisis+".csv")