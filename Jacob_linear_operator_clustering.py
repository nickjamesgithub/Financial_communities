import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from Utilities import dendrogram_plot

# Read in data
linear_operator_params = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/optimal_params.csv")
prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors_clean.csv",
                     index_col='Date')
# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$', '')
log_returns = np.log(prices).diff()[1:]


# Sector labels
sectors_labels = ["Health Care", "Industrials", "Information Technology", "Utilities", "Financials",
                  "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy",
                  "Communication Services"]
sectors_labels.sort()

# Map all periods to current financial crisis
dc_gfc_projection = linear_operator_params.iloc[1,:]
gfc_gfc_projection = linear_operator_params.iloc[5,:]
covid_gfc_projection = linear_operator_params.iloc[9,:]
ukraine_gfc_projection = linear_operator_params.iloc[13,:]

# Store sector/crisis/returns list
sector_crisis_returns_list = []
names_list = []
# Loop over all the sectors
for i in range(len(sectors_labels)):

    # Slice data with sector label
    sector_return_slice = log_returns[sectors_labels[i]]

    # Get all market periods
    sector_dc = sector_return_slice.iloc[146:794, :].mean(axis=1)
    sector_gfc = sector_return_slice.iloc[1865:2703, :].mean(axis=1)
    sector_covid = sector_return_slice.iloc[5184:5304, :].mean(axis=1)
    sector_ukraine = sector_return_slice.iloc[5642:5733, :].mean(axis=1)

    # Project all return periods into GFC (get intercept and slope parameters)
    # Dot Com
    alpha_dc = dc_gfc_projection.iloc[3]
    beta_dc = dc_gfc_projection.iloc[4]
    dc_projected = alpha_dc + beta_dc * sector_dc.values

    # GFC
    alpha_gfc = gfc_gfc_projection.iloc[3]
    beta_gfc = gfc_gfc_projection.iloc[4]
    gfc_projected = alpha_gfc + beta_gfc * sector_gfc.values

    # COVID
    alpha_covid = covid_gfc_projection.iloc[3]
    beta_covid = covid_gfc_projection.iloc[4]
    covid_projected = alpha_covid + beta_covid * sector_covid.values

    # Ukraine
    alpha_ukraine = ukraine_gfc_projection.iloc[3]
    beta_ukraine = ukraine_gfc_projection.iloc[4]
    ukraine_projected = alpha_ukraine + beta_ukraine * sector_ukraine.values

    # Append to list
    sector_crisis_returns_list.append(dc_projected)
    sector_crisis_returns_list.append(gfc_projected)
    sector_crisis_returns_list.append(covid_projected)
    sector_crisis_returns_list.append(ukraine_projected)

    # Append names list
    names_list.append(sectors_labels[i]+"_"+"Dot_com")
    names_list.append(sectors_labels[i] + "_" + "GFC")
    names_list.append(sectors_labels[i] + "_" + "Covid")
    names_list.append(sectors_labels[i] + "_" + "Ukraine")
    print("Sector iteration ", sectors_labels[i])

# Initialise distance matrix
distance_matrix = np.zeros((len(names_list),len(names_list)))
# Loop over all market periods
for i in range(len(sector_crisis_returns_list)):
    for j in range(len(sector_crisis_returns_list)):
        # Slice returns from stock i and stock j
        returns_i = sector_crisis_returns_list[i]
        returns_j = sector_crisis_returns_list[j]
        # Compute l^1 distance
        wasserstein = wasserstein_distance(returns_i,returns_j)
        distance_matrix[i,j] = wasserstein

# Plot Distance matrix
plt.matshow(distance_matrix)
plt.show()

