import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from turning_point_algorithm import identify_turning_points
from scipy.signal import savgol_filter

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean.csv", index_col='Date')
prices.dropna(axis='columns', inplace=True)

# Generate cryptocurrency returns
equity_returns = np.log(prices).diff()[1:]
# equity_returns.dropna(axis='columns', inplace=True)

# Smoothing Rate
smoothing_rate = 120

rolling_correlation_avg = []
# Smoothing Rate
for i in range(len(equity_returns)):
    # Returns
    returns = equity_returns.iloc[i - smoothing_rate:i, :]

    # Compute returns correlation with pandas
    correlation = np.nan_to_num(returns.corr())

    # Compute rolling correlation matrix norm
    correlation_norm = np.mean(correlation)
    rolling_correlation_avg.append(correlation_norm)

    print(i)

# Plot rolling correlation norm
fig, ax = plt.subplots()
date_index_plot = pd.date_range('05-01-2000','08-10-2020',len(rolling_correlation_avg)).strftime('%Y-%m-%d')
plt.plot(date_index_plot, rolling_correlation_avg, label="Rolling average correlation", color='black')
plt.tick_params(axis='x', which='major', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.xlabel("Time")
plt.ylabel("Rolling average market correlation")
plt.legend()
plt.savefig("Rolling_equities_correlation_norm")
plt.show()

