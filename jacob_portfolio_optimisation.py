import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh

crisis_list = ["dot_com", "gfc", "covid", "ukraine"]
crisis = "dot_com"
num_stocks = 40
# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_real.csv", index_col='Date')
prices.columns = prices.columns.str.replace('(\.\d+)$','')

# Clean prices
prices_clean = prices.iloc[:,1:]
prices_clean = prices_clean.dropna(axis=1)

# Slice sectors and stocks
sectors = prices_clean.columns
stocks = prices_clean.iloc[0,:].values

if crisis == "dot_com":
    prices = prices_clean.iloc[146:794,:]

if crisis == "gfc":
    prices = prices_clean.iloc[1865:2703,:]

if crisis == "covid":
    prices = prices_clean.iloc[5184:5304,:]

if crisis == "ukraine":
    prices = prices_clean.iloc[5642:5733,:]

# First pick n sectors at random
sector_sequence = list(np.linspace(0, len(prices.iloc[0]) - 1, len(prices.iloc[0])))  # Randomly draw sector numbers
random_list_sector = random.sample(sector_sequence, num_stocks)
ints = [int(item) for item in random_list_sector]

# Select random stocks
random_price = prices.iloc[:, ints] # Slice specific stocks
random_sectors = sectors[ints]
random_stocks = stocks[ints]

# Compute log returns of prices


# Store simulated values
sim_sharpe = []
sim_stocks = []
sim_sectors = []

