import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# Read in the data
df = pd.read_csv("/Users/tassjames/Desktop/optiver_assessment/final_data_10s.csv")

# Convert Time column to Date Time object
df["Time"] = pd.to_datetime(df["Time"])
df["X_Spread"] = df["X_BID"] - df["X_ASK"]
df["Y_Spread"] = df["Y_BID"] - df["Y_ASK"]

# Plot the two time series
plt.plot(df["Time"], df["X_BID"])
plt.plot(df["Time"], df["Y_BID"])
plt.xlabel("Time")
plt.ylabel("Bid Price")
plt.show()

# Plot the two time series
plt.plot(df["Time"], df["X_ASK"])
plt.plot(df["Time"], df["Y_ASK"])
plt.xlabel("Time")
plt.ylabel("Ask Price")
plt.show()

# Compute Log returns of X & Y
df['X_BID_log_returns'] = np.log(df["X_BID"]) - np.log(df["X_BID"].shift(1))
df['Y_BID_log_returns'] = np.log(df["Y_BID"]) - np.log(df["Y_BID"].shift(1))
df['X_ASK_log_returns'] = np.log(df["X_ASK"]) - np.log(df["X_ASK"].shift(1))
df['Y_ASK_log_returns'] = np.log(df["Y_ASK"]) - np.log(df["Y_ASK"].shift(1))

# # Plot distribution of log returns
# plt.hist(df["X_BID_log_returns"], bins=50, alpha = 0.4)
# plt.hist(df["Y_BID_log_returns"], bins=50, alpha = 0.4)
# plt.show()

x_mean = np.mean(df["X_BID_log_returns"])
x_std = np.var(df["X_BID_log_returns"])
y_mean = np.mean(df["Y_BID_log_returns"])
y_std = np.var(df["Y_BID_log_returns"])

# Compute static correlations
correlation_bid = df["X_BID_log_returns"].corr(df["Y_BID_log_returns"])
correlation_ask = df["X_ASK_log_returns"].corr(df["Y_ASK_log_returns"])

# Add additional features for minute, hour, day, week, month, year
df["Date"] = df["Time"].dt.date
df["Minute"] = df["Time"].dt.minute
df["Hour"] = df["Time"].dt.hour
df["Day"] = df["Time"].dt.day
df["Week"] = df["Time"].dt.week
df["Month"] = df["Time"].dt.month
df["Year"] = df["Time"].dt.year


# Write a function to generate a table over the space which may identify dislocations
def dislocation_function(df, grid_periodicity):
    # Loop over all grid components and assess divergence in correlation
    grid = np.linspace(np.min(df[grid_periodicity]), np.max(df[grid_periodicity]), len(np.unique(df[grid_periodicity])))
    metrics_periodicity_list = []
    for i in range(len(grid)):
        slice = df.loc[df[grid_periodicity] == grid[i]]
        bid_correlation = slice["X_BID_log_returns"].corr(slice["Y_BID_log_returns"])
        ask_correlation = slice["X_ASK_log_returns"].corr(slice["Y_ASK_log_returns"])
        X_spread = slice["X_Spread"].mean()
        Y_spread = slice["Y_Spread"].mean()
        x_returns_mean = slice["X_BID_log_returns"].mean()
        x_returns_sd = slice["X_BID_log_returns"].std()
        y_returns_mean = slice["Y_BID_log_returns"].mean()
        y_returns_sd = slice["Y_BID_log_returns"].std()
        returns_mean_diff = x_returns_mean - y_returns_mean
        metrics_periodicity_list.append([grid[i], bid_correlation, ask_correlation, X_spread, Y_spread, x_returns_mean,
                                         y_returns_mean, x_returns_sd, y_returns_sd, returns_mean_diff])

    # Correlation scores minute array
    metrics_periodicity_df = pd.DataFrame(metrics_periodicity_list)
    metrics_periodicity_df.columns = ["Periodicity", "Bid_correlation", "Ask_correlation", "X_Spread", "Y_Spread", "X_log_returns",
                                     "Y_log_returns", "X_standard_deviation", "Y_standard_deviation", "Returns_mean_difference"]
    return metrics_periodicity_df

def stationarity_test(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series is likely stationary.')
    else:
        print('p-value = ' + str(pvalue) + ' The series is likely non-stationary.')

# Loop over all minutes
# and assess divergence in correlation
dislocation_minute_grid = dislocation_function(df, "Minute")
dislocation_hour_grid = dislocation_function(df, "Hour")
dislocation_day_grid = dislocation_function(df, "Day")
dislocation_week_grid = dislocation_function(df, "Week")
dislocation_month_grid = dislocation_function(df, "Month")
dislocation_year_grid = dislocation_function(df, "Year")

# Study evolution of correlation throughout the 10am time series
# Loop over unique days within the 10am hour
date_unique = df["Date"].unique()
hour_unique = df["Hour"].unique()
minute_unique = df["Minute"].unique()

# Hour/minute grid
hour_spacing = np.linspace(np.min(hour_unique), np.max(hour_unique), len(hour_unique))
minute_spacing = np.linspace(np.min(minute_unique), np.max(minute_unique), len(minute_unique))

# Grouped data by hour and minute
grouped_hour_minute = df.groupby(["Hour", "Minute"]).agg({k: np.mean for k in ['X_BID_log_returns', 'Y_BID_log_returns', 'X_Spread', 'Y_Spread']})

# Data in each minute/hour increment
minute_hour_returns_list = []
for i in range(len(hour_spacing)):
    for j in range(len(minute_spacing)):
        unique_minute_hour = df.loc[(df["Minute"]==minute_spacing[j]) & (df["Hour"]==hour_spacing[i])]
        correlation_bid_xy = np.nan_to_num(unique_minute_hour["X_BID_log_returns"].corr(unique_minute_hour["Y_BID_log_returns"]))
        x_total_returns = unique_minute_hour["X_BID_log_returns"].sum()
        y_total_returns = unique_minute_hour["Y_BID_log_returns"].sum()
        x_avg_returns = unique_minute_hour["X_BID_log_returns"].sum()
        y_avg_returns = unique_minute_hour["Y_BID_log_returns"].sum()
        x_y_avg_mispricing = x_avg_returns - y_avg_returns
        x_y_total_mispricing = np.nan_to_num(x_total_returns - y_total_returns)
        minute_hour_returns_list.append([hour_spacing[i], minute_spacing[j], len(unique_minute_hour), correlation_bid_xy, x_avg_returns, y_avg_returns,
                                         x_total_returns, y_total_returns, x_y_avg_mispricing, x_y_total_mispricing])
# Convert to Dataframe
minute_hour_df = pd.DataFrame(minute_hour_returns_list)
minute_hour_df.columns = ["Hour", "Minute", "Samples", "Correlation", "X_avg_returns", "Y_avg_returns", "X_total_returns", "Y_total_returns", "X_Y_avg_mispricing", "X_Y_total_mispricing"]
minute_hour_df_sorted = minute_hour_df.sort_values(by="X_Y_avg_mispricing", ascending=False)

# Taking a look at anomolous moments... cannot seem to identify a temporal pattern viewing the data like this"

# On average, throughout the day - how do these stocks move?
x_returns = pd.DataFrame(minute_hour_df["X_avg_returns"])
y_returns = pd.DataFrame(minute_hour_df["Y_avg_returns"])
x_cleaned_returns = x_returns.fillna(0)
y_cleaned_returns = y_returns.fillna(0)

# Key points of time:
# 8:00am Y outperforms X 3 times magnitude positive
# 10:00am Y underperforms X 2 times magnitude negative
# 11:00am - 11:03am X appears to underperform Y in a recalibration

# Partition before and after 10 o'clock in case of local stationarity / non-stationarity
x_first_partition = x_cleaned_returns[0:120]
y_first_partition = y_cleaned_returns[0:120]
x_second_partition = x_cleaned_returns[180:360]
y_second_partition = y_cleaned_returns[180:360]

# Compute Stationarity for both instruments X & Y before and after 10am structural break
stationarity_test(x_first_partition)
stationarity_test(y_first_partition)
stationarity_test(x_second_partition)
stationarity_test(y_second_partition)

# Test for correlation and cointegration
print('Correlation: ' + str(np.corrcoef(np.array(x_cleaned_returns["X_avg_returns"]), np.array(y_cleaned_returns["Y_avg_returns"]))))
score, pvalue, _ = coint(np.nan_to_num(x_cleaned_returns["X_avg_returns"]), np.nan_to_num(y_cleaned_returns["Y_avg_returns"]))
print('Cointegration test p-value: ' + str(pvalue))

# The series appear to be both correlated and cointegrated
# Compute Cumulative returns
x_avg_cum_returns = x_cleaned_returns.cumsum()
y_avg_cum_returns = y_cleaned_returns.cumsum()

# Plot of cumulative log returns for X vs Y
plt.plot(x_avg_cum_returns)
plt.plot(y_avg_cum_returns)
plt.xlabel("Minutes throughout the day")
plt.ylabel("Avg cumulative returns")
plt.show()

# Plot deviation between average returns for 2 instruments: X & Y
plt.plot(x_cleaned_returns["X_avg_returns"] - y_cleaned_returns["Y_avg_returns"])# Plot the spread between average returns
plt.axhline((x_cleaned_returns["X_avg_returns"] - y_cleaned_returns["Y_avg_returns"]).mean(), color='red', alpha = 0.5, linestyle='--') # Add the mean
plt.xlabel('Time')
plt.legend(['Log Returns Spread (X-Y)', 'Mean'])
plt.show()

x=1

