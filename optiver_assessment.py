import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

conditions = [
    (df['X_BID_log_returns'] == 0) & (df['Y_BID_log_returns'] == 0),
    (df['X_BID_log_returns'] > 0) & (df['Y_BID_log_returns'] > 0),
    (df['X_BID_log_returns'] > 0) & (df['Y_BID_log_returns'] < 0),
    (df['X_BID_log_returns'] < 0) & (df['Y_BID_log_returns'] > 0),
    (df['X_BID_log_returns'] < 0) & (df['Y_BID_log_returns'] < 0)
    ]

# create a list of the values we want to assign for each condition
values = ['NULL', 'LONG_X_LONG_Y', 'LONG_X_SHORT_Y', 'SHORT_X_LONG_Y', 'SHORT_X_SHORT_Y']

# create a new column and use np.select to assign values to it using our lists as arguments
df['TRADE'] = np.select(conditions, values)

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
        metrics_periodicity_list.append([grid[i], bid_correlation, ask_correlation, X_spread, Y_spread])

    # Correlation scores minute array
    metrics_periodicity_array = np.array(metrics_periodicity_list)

    return metrics_periodicity_array

# Loop over all minutes
# and assess divergence in correlation
minute_grid = dislocation_function(df, "Minute")
hour_grid = dislocation_function(df, "Hour")
day_grid = dislocation_function(df, "Day")
week_grid = dislocation_function(df, "Week")
month_grid = dislocation_function(df, "Month")
year_grid = dislocation_function(df, "Year")

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
minute_hour_list = []
for i in range(len(hour_spacing)):
    for j in range(len(minute_spacing)):
        unique_minute_hour = df.loc[(df["Minute"]==minute_spacing[j]) & (df["Hour"]==hour_spacing[i])]
        correlation_bid_xy = np.nan_to_num(unique_minute_hour["X_BID_log_returns"].corr(unique_minute_hour["Y_BID_log_returns"]))
        correlation_ask_xy = np.nan_to_num(unique_minute_hour["X_ASK_log_returns"].corr(unique_minute_hour["Y_ASK_log_returns"]))
        minute_hour_list.append([hour_spacing[i], minute_spacing[j], len(unique_minute_hour), correlation_bid_xy, correlation_ask_xy])

minute_hour_array = np.array(minute_hour_list)
x=1



