import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from Utilities import dendrogram_plot_test

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
# Sector abbreviated
sector_abbreviated_labels = ["Health Care", "Industrials", "IT", "Utilities", "Financials",
                  "Materials", "Real Estate", "C. Staples", "C. Discretionary", "Energy",
                  "Comms"]
sectors_labels.sort()

# Map all periods to current financial crisis
dc_gfc_projection = linear_operator_params.iloc[1,:] # 1
gfc_gfc_projection = linear_operator_params.iloc[5,:] # 5
covid_gfc_projection = linear_operator_params.iloc[9,:] # 9
ukraine_gfc_projection = linear_operator_params.iloc[13,:] # 13

# Store sector/crisis/returns list
sector_crisis_returns_list = []
names_list = []
# Loop over all the sectors
for i in range(len(sectors_labels)):

    # Slice data with sector label
    sector_return_slice = 100 * log_returns[sectors_labels[i]]

    # Get all market periods
    sector_dc = sector_return_slice.iloc[146:794, :].values.flatten()
    sector_gfc = sector_return_slice.iloc[1865:2703, :].values.flatten()
    sector_covid = sector_return_slice.iloc[5184:5304, :].values.flatten()
    sector_ukraine = sector_return_slice.iloc[5642:5733, :].values.flatten()

    # Project all return periods into GFC (get intercept and slope parameters)
    # Dot Com
    alpha_dc = dc_gfc_projection.iloc[3]
    beta_dc = dc_gfc_projection.iloc[4]
    dc_projected = alpha_dc + beta_dc * sector_dc #.values

    # GFC
    alpha_gfc = gfc_gfc_projection.iloc[3]
    beta_gfc = gfc_gfc_projection.iloc[4]
    gfc_projected = alpha_gfc + beta_gfc * sector_gfc #.values

    # COVID
    alpha_covid = covid_gfc_projection.iloc[3]
    beta_covid = covid_gfc_projection.iloc[4]
    covid_projected = alpha_covid + beta_covid * sector_covid #.values

    # Ukraine
    alpha_ukraine = ukraine_gfc_projection.iloc[3]
    beta_ukraine = ukraine_gfc_projection.iloc[4]
    ukraine_projected = alpha_ukraine + beta_ukraine * sector_ukraine #.values

    # Append to list
    sector_crisis_returns_list.append(dc_projected)
    sector_crisis_returns_list.append(gfc_projected)
    sector_crisis_returns_list.append(covid_projected)
    sector_crisis_returns_list.append(ukraine_projected)

    # Append names list
    names_list.append(sector_abbreviated_labels[i]+"_"+"Dot_com")
    names_list.append(sector_abbreviated_labels[i] + "_" + "GFC")
    names_list.append(sector_abbreviated_labels[i] + "_" + "Covid")
    names_list.append(sector_abbreviated_labels[i] + "_" + "Ukraine")
    print("Sector iteration ", sector_abbreviated_labels[i])

# Initialise distance matrix
distance_matrix = np.zeros((len(names_list),len(names_list)))
# Loop over all market periods
for i in range(len(sector_crisis_returns_list)):
    for j in range(len(sector_crisis_returns_list)):
        # Slice returns from stock i and stock j
        returns_i = sector_crisis_returns_list[i]
        returns_j = sector_crisis_returns_list[j]
        # Compute Wasserstein distance
        wasserstein = wasserstein_distance(returns_i,returns_j)
        distance_matrix[i,j] = wasserstein

# Plot Distance matrix
plt.matshow(distance_matrix)
plt.show()

# Create dendrogram plot
dendrogram_plot_test(distance_matrix, "wasserstein_", "sector_returns_crisis", names_list)

