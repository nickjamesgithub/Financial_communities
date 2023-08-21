import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
np.random.seed(123)

crisis_list = ["dot_com", "gfc", "covid", "ukraine"]
num_stocks = 40

for c in range(len(crisis_list)):
    crisis = crisis_list[c]

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

    if crisis == "dot_com":
        prices = prices_clean.iloc[146:794,:].astype("float")

    if crisis == "gfc":
        prices = prices_clean.iloc[1865:2703,:].astype("float")

    if crisis == "covid":
        prices = prices_clean.iloc[5184:5304,:].astype("float")

    if crisis == "ukraine":
        prices = prices_clean.iloc[5642:5733,:].astype("float")

    # Store simulated values
    sim_sharpe = []
    sim_stocks = []
    sim_sectors = []

    while len(sim_sharpe) < 250000:
        # First pick n sectors at random
        sector_sequence = list(np.linspace(0, len(prices_clean.iloc[0]) - 1, len(prices_clean.iloc[0])))  # Randomly draw sector numbers
        random_list_sector = random.sample(sector_sequence, num_stocks)
        ints = [int(item) for item in random_list_sector]

        # Select random stocks
        random_price = prices_clean.iloc[:, ints] # Slice specific stocks
        random_sectors = sectors[ints]
        random_stocks = stocks[ints]

        # Compute log returns of prices
        log_returns = np.log(random_price).diff()[1:]
        total_returns = np.sum(log_returns, axis=0)
        cov = np.cov(np.transpose(log_returns))
        weights = np.repeat(1/len(random_stocks), len(random_stocks))# .reshape(len(random_stocks),1)

        # Compute portfolio returns
        portfolio_returns = np.matmul(weights, total_returns)
        portfolio_std = np.dot(np.dot(np.transpose(weights), cov), weights) * np.sqrt(250)
        portfolio_sharpe = portfolio_returns/portfolio_std

        # Append Sharpe, stocks and sector
        sim_sharpe.append(portfolio_sharpe) # Append Sharpe Ratio
        sim_sectors.append(random_sectors) # Append Sectors
        sim_stocks.append(random_stocks) # Append Stocks
        print("Iteration ", len(sim_sharpe))

# Convert to dataframes & merge dataframes
sim_sharpe_df = pd.DataFrame(sim_sharpe)
sim_stocks_df = pd.DataFrame(sim_stocks)
sim_sectors_df = pd.DataFrame(sim_sectors)

# Merged Dataframe
merged_df = pd.concat([sim_sharpe_df, sim_stocks_df, sim_sectors_df], axis=1)
merged_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/portfolio_simulation_results_"+crisis+".csv")


