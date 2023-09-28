import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import datetime
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

surface = pd.read_csv("/Users/tassjames/Desktop/Continuous_Sharpe.csv")

# plot surface
plt.matshow(np.array(surface))
plt.show()

# Import data
data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/portfolio_optimisation_k_yearly.csv")
data.columns = ["Index", "Time", "portfolio_k", "decile_10", "decile_50", "decile_90"]

# Set up initial parameters
idx_length = 252
upper_bound = 5796
time_grid = np.arange(idx_length, upper_bound, idx_length)
portfolio_lb = 10
portfolio_ub = 100
k_grid = np.arange(portfolio_lb,portfolio_ub,1)

# Loop over time grid
grid_surface_90 = []
for i in range(len(time_grid)):
    print("Test ", time_grid[i])
    # Slice time point
    data_time_slice = data.loc[data["Time"]==time_grid[i]]
    # Loop over k
    sharpes_list_90 = []
    sharpes_list_50 = []
    sharpes_list_10 = []
    for k in range(len(k_grid)):
        # Slice for portfolio of size K
        print("Portfolio size ", k_grid[k])
        data_time_k_slice = data_time_slice.loc[data_time_slice["portfolio_k"]==k_grid[k]]
        # Append decile plots for various sharpe ratios
        sharpes_list_90.append(data_time_k_slice["decile_90"].iloc[0])
        sharpes_list_50.append(data_time_k_slice["decile_50"].iloc[0])
        sharpes_list_10.append(data_time_k_slice["decile_10"].iloc[0])

    # Plot incremental Sharpe Ratio
    grid = np.linspace(10,len(sharpes_list_90),90)
    plt.plot(grid, sharpes_list_10, label="10th percentile", alpha = 0.5)
    plt.plot(grid, sharpes_list_50, label="50th percentile", alpha = 0.5)
    plt.plot(grid, sharpes_list_90, label="90th percentile", alpha = 0.5)
    plt.xlabel("Portfolio size")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.title("Time_period_" +str(time_grid[i]))
    plt.savefig("Portfolio_diversification_size_percentile_"+str(time_grid[i]))
    plt.show()

    # Repeat surface and reshape
    sharpes_list_90_surface = np.repeat([sharpes_list_90], idx_length, axis=0)
    sharpes_surface_90 = np.reshape(sharpes_list_90_surface, (idx_length, (portfolio_ub-portfolio_lb)))
    grid_surface_90.append(sharpes_surface_90)
