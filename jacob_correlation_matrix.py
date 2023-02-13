import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Helper functions
def flatten_dataframe(dataframe):
    df_out = dataframe.corr().stack()
    df_out = df_out[df_out.index.get_level_values(0) != df_out.index.get_level_values(1)]
    df_out.index = df_out.index.map('_'.join)
    df_out = df_out.to_frame().T
    return df_out

# Import clean dataset
data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors_clean.csv")
data = data.iloc[1:,2:]

# Compute log returns of dataset
log_returns = np.log(data).diff()[1:]

# Slice data for specific dates of financial crises
dotcom_returns = log_returns.iloc[146:794,:]
gfc_returns = log_returns.iloc[1865:2703,:]
gfc_peak_returns = log_returns.iloc[2284:2471,:]
covid_returns = log_returns.iloc[5184:5304,:]
covid_peak_returns = log_returns.iloc[5184:5239,:]
ukraine_crash_returns = log_returns.iloc[5642:5733,:]

# Correlation matrices for different periods
dotcom_corr = dotcom_returns.corr()

# Plot Correlation coefficients from all periods
# DotCom distribution
dotcom_values = flatten_dataframe(dotcom_returns)
dotcom_values_array = np.array(dotcom_values).flatten()

# GFC distribution
gfc_values = flatten_dataframe(gfc_returns)
gfc_values_array = np.array(gfc_values).flatten()

# GFC Peak distribution
gfc_peak_values = flatten_dataframe(gfc_peak_returns)
gfc_peak_values_array = np.array(gfc_peak_values).flatten()

# COVID distribution
covid_values = flatten_dataframe(covid_returns)
covid_values_array = np.array(covid_values).flatten()

# COVID Peak distribution
covid_peak_values = flatten_dataframe(covid_peak_returns)
covid_peak_values_array = np.array(covid_peak_values).flatten()

# Ukraine distribution
ukraine_crash_values = flatten_dataframe(ukraine_crash_returns)
ukraine_crash_values_array = np.array(ukraine_crash_values).flatten()

# Histogram plots
plt.hist(dotcom_values_array, bins=100, alpha=0.2, label='Dot-com')
plt.hist(gfc_values_array, bins=100, alpha=0.2, label='GFC')
plt.hist(gfc_peak_values_array, bins=100, alpha=0.2, label='GFC Peak')
plt.hist(covid_values_array, bins=100, alpha=0.2, label='COVID')
plt.hist(covid_peak_values_array, bins=100, alpha=0.2, label='COVID Peak')
plt.hist(ukraine_crash_values_array, bins=100, alpha=0.55, label='2022 Crash', color='red')
plt.legend()
plt.ylabel("Frequency")
plt.xlabel("Correlation coefficient")
plt.title("Correlation coefficient distribution")
plt.savefig("Correlation_coefficient_distribution_2022")
plt.show()
