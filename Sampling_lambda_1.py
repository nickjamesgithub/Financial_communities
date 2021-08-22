import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random

# Choose number of sectors and n for simulation
num_sectors = 5
n = 35
sample_per_sector = n//num_sectors

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/data/prices_sector_labels.csv", index_col='Date')

sectors_labels = ["#N/A", "Health Care", "Industrials", "Communication Services", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy"]
sectors_labels.sort()

# First pick n sectors at random
sector_sequence = list(np.linspace(1,11,11))
random_list = random.sample(sector_sequence, 5)
ints = [int(item) for item in random_list]

# Get corresponding sector names
random_sector_list = []
for i in range(len(ints)):
    random_sector_drawn = sectors_labels[ints[i]]
    random_sector_list.append(random_sector_drawn)

# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$','')
prices_df = pd.DataFrame(prices)

# Loop over all sectors and get column indices


# Get the random samples for current iteration
stock_samples = []
for i in range(len(random_sector_list)):
    # index_no = prices_df.columns.get_loc(random_sector_list[i])
    sector_slice = prices_df[random_sector_list[i]]
    length = len(sector_slice.iloc[0,:])
    random_sequence = list(np.linspace(1, length-1, length-1))
    random_list = random.sample(sector_sequence, sample_per_sector)
    ints = [int(item) for item in random_list]
    random_sector_stocks = sector_slice.iloc[:,ints]
    stock_samples.append(random_sector_stocks)

block = 1