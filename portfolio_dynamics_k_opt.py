import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
np.random.seed(123)

stock_list = np.linspace(10,60,51)
portfolio_combination = []

for i in range(len(stock_list)):
    # Slice a number of stocks
    num_stocks = np.int(stock_list[i])

    # Import data
    prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_real.csv", index_col='Date')
    prices.columns = prices.columns.str.replace('(\.\d+)$','')
    v_counts = prices.columns.value_counts()

    # Clean prices
    prices_clean = prices.iloc[:,1:]
    prices_clean = prices_clean.dropna(axis=1)

    # Slice sectors and stocks
    sectors = prices_clean.columns
    stocks = prices_clean.iloc[0,:].values

    # Store simulated values
    sim_sharpe = []
    sim_stocks = []
    sim_sectors = []
    while len(sim_sharpe) < 1000:
        # First pick n sectors at random
        sector_sequence = list(np.linspace(0, len(prices_clean.iloc[0]) - 1, len(prices_clean.iloc[0])))  # Randomly draw sector numbers
        random_list_sector = random.sample(sector_sequence, num_stocks)
        ints = [int(item) for item in random_list_sector]

        # Select random stocks
        random_price = prices_clean.iloc[1:, ints] # Slice specific stocks
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
        portfolio_std = np.dot(np.dot(np.transpose(weights), cov), weights)
        portfolio_sharpe = portfolio_returns/portfolio_std

        # Append Sharpe, stocks and sector
        sim_sharpe.append(portfolio_sharpe) # Append Sharpe Ratio
        sim_sectors.append(random_sectors) # Append Sectors
        sim_stocks.append(random_stocks) # Append Stocks
        print("Iteration ", len(sim_sharpe))

    # Compute top decile
    decile = np.percentile(sim_sharpe, 10)
    x=1
    y=2