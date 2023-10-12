import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
import statsmodels.api as sm

#todo NOTE LABELLING IS HARDCODED IN THE PLOT

make_plots = True
model_choice = "decile_10"

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
model1_params = []
model2_params = []
for i in range(len(time_grid)):
    print("Test ", time_grid[i])
    # Slice time point
    data_time_slice = data.loc[data["Time"]==time_grid[i]]
    # Get sharpe ratios
    decile_90 = data_time_slice["decile_90"]
    decile_50 = data_time_slice["decile_50"]
    decile_10 = data_time_slice["decile_10"]

    # Slice response variable and two predictors
    #todo this is where Y comes in
    if model_choice == "decile_10":
        y = np.array(decile_10).reshape(-1, 1)
    if model_choice == "decile_50":
        y = np.array(decile_50).reshape(-1, 1)
    if model_choice == "decile_90":
        y = np.array(decile_90).reshape(-1, 1)
    x1 = np.reshape(np.linspace(10, 100, 91), (len(decile_90), 1))  # Linear
    x1_ones = sm.tools.tools.add_constant(x1)
    x2 = np.reshape(np.linspace(10, 100, 91), (len(decile_90), 1))**2  # Quadratic

    # Combinations of features
    linear_quadratic = np.concatenate((x1, x2), axis=1)  # linear + indicator

    # Add column of ones
    linear_ones = sm.tools.tools.add_constant(x1)
    linear_quadratic_ones = sm.tools.tools.add_constant(linear_quadratic)

    # Model 1 statsmodels: linear
    model1 = sm.OLS(y, linear_ones)
    results1 = model1.fit()
    # AIC/BIC/Adjusted R2
    m1_params = results1.params
    m1_aic = results1.aic
    m1_bic = results1.bic
    m1_r2a = results1.rsquared_adj
    m1_pvals = results1.pvalues
    # Append parameters to Model 1 list
    model1_params.append([time_grid[i], m1_params, m1_aic, m1_bic, m1_r2a, m1_pvals])

    # Model 1 statsmodels: linear + Quadratic
    model2 = sm.OLS(y, linear_quadratic_ones)
    results2 = model2.fit()
    # AIC/BIC/Adjusted R2
    m2_params = results2.params
    m2_aic = results2.aic
    m2_bic = results2.bic
    m2_r2a = results2.rsquared_adj
    m2_pvals = results2.pvalues
    # Append parameters to Model 2 list
    model2_params.append([time_grid[i], m2_params, m2_aic, m2_bic, m2_r2a, m2_pvals])

    if make_plots:
        # Model 1, Model 2, Model 3, Model 4 fit (statsmodels)
        fig, ax = plt.subplots()
        grid = np.linspace(10,100,91)
        plt.plot(grid, results1.fittedvalues, label="Model 1", alpha=0.6)
        plt.plot(grid, results2.fittedvalues, label="Model 2", alpha=0.6)
        plt.scatter(grid, y, label="data", color='red')
        plt.locator_params(axis='x', nbins=4)
        plt.ylabel("Sharpe Ratio")
        plt.xlabel("Portfolio_size")
        plt.legend()
        # plt.title(events_list_m[i]+"_"+gender_labels[g] + "_" + str(top))
        plt.savefig("Portfolio_size_regression"+model_choice+"_"+str(time_grid[i]))
        plt.show()

# Model parameters
m1_params_df = pd.DataFrame(model1_params)
m2_params_df = pd.DataFrame(model2_params)
m1_params_df.columns = ["Time", "Parameters", "AIC", "BIC", "Adjusted_R2", "p_values"]
m2_params_df.columns = ["Time", "Parameters", "AIC", "BIC", "Adjusted_R2", "p_values"]
m1_params_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/model_1_params_"+model_choice+".csv")
m2_params_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/model_2_params_"+model_choice+".csv")
