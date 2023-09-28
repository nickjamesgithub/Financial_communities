import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
np.random.seed(123)

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_real.csv", index_col='Date')
prices.columns = prices.columns.str.replace('(\.\d+)$', '')
v_counts = prices.columns.value_counts()

# Clean prices
prices_clean = prices.iloc[106:, 1:]
prices_clean = prices_clean.dropna(axis=1)

# initialise smoothing rate
block_rate = 252
portfolio_lb = 10
portfolio_ub = 100
portfolio_mix = portfolio_ub - portfolio_lb + 1
length_simulation = 5796
num_simulations = 1000 # 1000
portfolio_combination_list = []

for s in range(block_rate, length_simulation, block_rate): # len(prices_clean)
    # Slice Prices
    prices_clean_slice = prices_clean.iloc[(s - block_rate):s, :]

    # Generate stock list & Initialise portfolio combination list
    stock_list = np.linspace(portfolio_lb,portfolio_ub,portfolio_mix)

    # Loop over stock combinations
    for i in range(len(stock_list)):
        # Slice a number of stocks
        num_stocks_iterate = np.int(stock_list[i])

        # Slice sectors and stocks
        sectors = prices_clean_slice.columns
        stocks = prices_clean_slice.iloc[0,:].values

        # Store simulated values
        sim_sharpe = []
        sim_stocks = []
        sim_sectors = []
        while len(sim_sharpe) < num_simulations:
            # First pick n sectors at random
            sector_sequence = list(np.linspace(0, len(prices_clean_slice.iloc[0]) - 1, len(prices_clean_slice.iloc[0])))  # Randomly draw sector numbers
            random_list_sector = random.sample(sector_sequence, num_stocks_iterate)
            ints = [int(item) for item in random_list_sector]

            # Select random stocks
            random_price = prices_clean_slice.iloc[1:, ints] # Slice specific stocks
            random_price = random_price.astype("float")
            random_sectors = sectors[ints]
            random_stocks = stocks[ints]

            # Compute log returns of prices
            log_returns = np.log(random_price).diff()[1:]
            total_returns = np.sum(log_returns, axis=0)
            avg_returns = np.mean(log_returns, axis=0)
            cov = np.cov(np.transpose(log_returns))
            weights = np.repeat(1/len(random_stocks), len(random_stocks))# .reshape(len(random_stocks),1)

            # Compute portfolio returns
            portfolio_returns = np.dot(weights, avg_returns)
            portfolio_std = np.dot(np.dot(np.transpose(weights), cov), weights) * np.sqrt(250)
            portfolio_sharpe = portfolio_returns/portfolio_std

            # Append Sharpe, stocks and sector
            sim_sharpe.append(portfolio_sharpe) # Append Sharpe Ratio
            sim_sectors.append(random_sectors) # Append Sectors
            sim_stocks.append(random_stocks) # Append Stocks

            # Simulation update
            print("Block-start: ", s - block_rate)
            print("Block-end: ", s)
            print("Simulation: ", i)
            print("Iteration ", len(sim_sharpe))

        # Compute deciles and portfolio specifics
        p_10 = np.percentile(sim_sharpe, 10)
        p_50 = np.percentile(sim_sharpe, 50)
        p_90 = np.percentile(sim_sharpe, 90)
        portfolio_combination_list.append([s, num_stocks_iterate, p_10, p_50, p_90])

# Make portfolio combination list a Dataframe
portfolio_combination_df = pd.DataFrame(portfolio_combination_list)
portfolio_combination_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/portfolio_optimisation_k_yearly.csv")

