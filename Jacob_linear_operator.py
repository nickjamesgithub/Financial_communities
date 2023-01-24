import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors_clean.csv", index_col='Date')
# Replace column names for prices
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$', '')

# Get unique column names for sectors
unique_columns = prices.columns.unique()

# Write out all the slices for various economic periods
dot_com = prices.iloc[146:794, :]
gfc = prices.iloc[1865:2703, :]
covid = prices.iloc[5184:5304, :]
ukraine = prices.iloc[5642:5733, :]

# Log returns of all economic periods
dot_com_returns = 100 * np.log(dot_com).diff()[1:]
gfc_returns = 100 * np.log(gfc).diff()[1:]
covid_returns = 100 * np.log(covid).diff()[1:]
ukraine_returns = 100 * np.log(ukraine).diff()[1:]

# Flatten probability distributions
dc_flattened = dot_com_returns.values.flatten()
gfc_flattened = gfc_returns.values.flatten()
covid_flattened = covid_returns.values.flatten()
ukraine_flattened = ukraine_returns.values.flatten()

# Store all the Flattened distributions in list
flattened_dist_names = ["Dot-Com", "GFC", "COVID", "Ukraine"]
flattened_distributions = [dc_flattened, gfc_flattened, covid_flattened, ukraine_flattened]
# Initialise alpha and beta values
alpha = np.linspace(-10, 10, 50)
beta = np.linspace(1/4, 4, 50)

# Optimise for alpha and beta parameter
optimal_parameter_list = []
for i in range(len(flattened_distributions)):
    for j in range(len(flattened_distributions)):
        print("Distribution", i)
        print("Distribution", j)
        dist_i = flattened_distributions[i]
        dist_j = flattened_distributions[j]
        # Loop over alpha and beta
        wasserstein_dist_list = []
        for a in range(len(alpha)):
            for b in range(len(beta)):
                print("Alpha parameter", a)
                print("Beta parameter", b)
                # Transformed distribution i
                dist_i_transformed = alpha[a] + beta[b] * dist_i
                # Compute Wasserstein distance
                wass_dist = wasserstein_distance(dist_i_transformed, dist_j)
                wasserstein_dist_list.append([flattened_dist_names[i], flattened_dist_names[j], alpha[a], beta[b], wass_dist])

        # Wasserstein array
        wasserstein_distance_array = np.array(wasserstein_dist_list)
        argmin_value = wasserstein_distance_array[:, 4].argmin()
        optimal_params = wasserstein_distance_array[argmin_value]
        # Append to optimal parameter list
        optimal_parameter_list.append(optimal_params)

# Write to csv file
optimal_parameter_df = pd.DataFrame(optimal_parameter_list)
optimal_parameter_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/optimal_params.csv")


# Correlation matrix for each period
dot_com_correlation = dot_com_returns.corr()
gfc_correlation = gfc_returns.corr()
covid_correlation = covid_returns.corr()
ukraine_correlation = ukraine_returns.corr()

# Plot heatmaps
# Dotcom
plt.matshow(dot_com_correlation)
plt.title("Dot com")
plt.colorbar()
plt.show()

# GFC
plt.matshow(gfc_correlation)
plt.title("GFC")
plt.colorbar()
plt.show()

# covid-19
plt.matshow(covid_correlation)
plt.title("COVID-19")
plt.colorbar()
plt.show()

# Ukraine
plt.matshow(ukraine_correlation)
plt.title("Ukraine Crisis")
plt.colorbar()
plt.show()



