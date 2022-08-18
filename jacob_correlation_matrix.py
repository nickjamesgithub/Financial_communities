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
dotcom_values = flatten_dataframe(gfc_returns)
dotcom_values_array = np.array(dotcom_values).flatten()

# Histogram plots
plt.hist(dotcom_values_array, bins=30, alpha=0.25, label='Dot-com bubble')
plt.title("Dot-com bubble")
plt.show()
