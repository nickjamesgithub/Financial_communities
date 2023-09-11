import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
import statsmodels.api as sm

make_plots = False

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_real.csv", index_col='Date')
prices.columns = prices.columns.str.replace('(\.\d+)$', '')
date_index = prices.index

# Import data
data = pd.read_csv("/Users/tassjames/Desktop/portfolio_optimisation_k.csv")
data.columns = ["Index", "Time", "portfolio_k", "decile_10", "decile_50", "decile_90"]

# Set up initial parameters
idx_length = 250
upper_bound = 5500
time_grid = np.arange(idx_length, upper_bound, idx_length)
portfolio_lb = 10
portfolio_ub = 100
k_grid = np.arange(portfolio_lb,portfolio_ub,1)

# Loop over time grid
optimal_cardinality_list = []
for i in range(len(time_grid)):
    print("Test ", time_grid[i])
    # Slice time point
    data_time_slice = data.loc[data["Time"]==time_grid[i]]
    # Get sharpe ratios
    decile_10 = data_time_slice["decile_10"]

    # Estimate beta for regression coefficient
    # Slice response variable and reformat for data structure
    y = np.array(decile_10).reshape(-1, 1)
    x1 = np.reshape(np.linspace(10, 100, 91), (len(decile_10), 1))  # Linear
    x1_ones = sm.tools.tools.add_constant(x1)

    # Add column of ones
    linear_ones = sm.tools.tools.add_constant(x1)

    # Model 1 statsmodels: linear
    model1 = sm.OLS(y, linear_ones)
    results1 = model1.fit()
    # Regression coefficient
    beta = results1.params[1]
    print("Regression coefficient ", beta, " ", time_grid[i])

    # Penalty Sharpe Ratio
    penalty = 0
    penalty_grid = np.linspace(portfolio_lb, portfolio_ub, portfolio_ub-portfolio_lb)
    penalised_sharpe_list = []
    for p in range(len(penalty_grid)):
        sharpe_penalty = decile_10.iloc[p] - penalty_grid[p] * beta
        penalised_sharpe_list.append(sharpe_penalty)
    # Compute Argmax
    raw_argmax = np.argmax(decile_10)
    penalised_argmax = np.argmax(penalised_sharpe_list)
    optimal_cardinality_list.append([time_grid[i], date_index[time_grid[i]], raw_argmax, penalised_argmax, np.array(decile_10)[raw_argmax], np.array(decile_10)[penalised_argmax]])

# Optimal cardinality dataframe
optimal_cardinality_df = pd.DataFrame(optimal_cardinality_list)
optimal_cardinality_df.columns = ["Time", "Date_index", "Unpenalised_argmax", "Penalised_argmax", "Sharpe_unpenalised_max", "Sharpe_penalised_max"]
optimal_cardinality_df.to_csv("/Users/tassjames/Desktop/portfolio_optimisation_k_penalty.csv")

# Generate deviation column
optimal_cardinality_df["Deviation"] = optimal_cardinality_df["Sharpe_unpenalised_max"] - optimal_cardinality_df["Sharpe_penalised_max"]

# Plot deviation vs Unpenalised Maximum
plt.scatter(optimal_cardinality_df["Sharpe_unpenalised_max"], optimal_cardinality_df["Deviation"],
            color='blue', alpha=0.5)
plt.xlabel("Sharpe unpenalised maximum")
plt.ylabel("Penalised deviation")
plt.savefig("Portfolio_dynamics_k_penalised_deviation")
plt.show()

