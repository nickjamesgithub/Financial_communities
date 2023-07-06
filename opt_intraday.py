import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from scipy.signal import savgol_filter
from scipy.stats import wasserstein_distance
from scipy.signal import welch
import datetime
from scipy.fftpack import fft, fftshift
from scipy.spatial.distance import euclidean

make_plots = True

# Read in the data
df = pd.read_csv("/Users/tassjames/Desktop/optiver_assessment/final_data_10s.csv")

# Convert Time column to Date Time object
df["Time"] = pd.to_datetime(df["Time"])
df["X_Spread"] = df["X_BID"] - df["X_ASK"]
df["Y_Spread"] = df["Y_BID"] - df["Y_ASK"]

if make_plots:
    # Plot the two time series
    plt.plot(df["Time"], df["X_BID"])
    plt.plot(df["Time"], df["Y_BID"])
    plt.xlabel("Time")
    plt.ylabel("Bid Price")
    plt.savefig("Opt_Bid_TS")
    plt.show()

    # Plot the two time series
    plt.plot(df["Time"], df["X_ASK"])
    plt.plot(df["Time"], df["Y_ASK"])
    plt.xlabel("Time")
    plt.ylabel("Ask Price")
    plt.savefig("Opt_Ask_TS")
    plt.show()

    # Plot the two time series
    plt.plot(df["Time"], df["X_Spread"], label="X_liquidity")
    plt.plot(df["Time"], df["Y_Spread"], label="Y_liquidity")
    plt.xlabel("Time")
    plt.ylabel("Bid-Ask: X & Y")
    plt.savefig("Opt_XY_Liquidity")
    plt.show()

# Compute Log returns of X & Y
df['X_BID_log_returns'] = np.log(df["X_BID"]) - np.log(df["X_BID"].shift(1))
df['Y_BID_log_returns'] = np.log(df["Y_BID"]) - np.log(df["Y_BID"].shift(1))
df['X_ASK_log_returns'] = np.log(df["X_ASK"]) - np.log(df["X_ASK"].shift(1))
df['Y_ASK_log_returns'] = np.log(df["Y_ASK"]) - np.log(df["Y_ASK"].shift(1))

# Study power spectrum
x_returns = np.nan_to_num(df["X_BID_log_returns"])
y_returns = np.nan_to_num(df["Y_BID_log_returns"])
x_returns_demean = x_returns - np.mean(x_returns)
y_returns_demean = y_returns - np.mean(y_returns)

# Plot power spectrum
log_periodogram_x = np.log(np.abs((fft(fftshift(x_returns_demean)))**2))[1:len(y_returns_demean)//2]
log_periodogram_y = np.log(np.abs((fft(fftshift(y_returns_demean)))**2))[1:len(y_returns_demean)//2]

# Parameters
window_size = (6 * 60 * 8) + 1 # Try various smoothing windows corresponding to 3 hours, 8 hours, etc.
polynomial_order = 2
X_spectrum_smooth = np.nan_to_num(savgol_filter(log_periodogram_x, window_size, polynomial_order))
Y_spectrum_smooth = np.nan_to_num(savgol_filter(log_periodogram_y, window_size, polynomial_order))

# Plot power spectrum - X
plt.plot(np.linspace(0, 0.5, (len(log_periodogram_x))), X_spectrum_smooth, label="X-log-periodogram", alpha=0.25)
plt.legend()
plt.savefig("opt_securities_power_spectrum_X")
plt.show()

# Plot power spectrum - Y
plt.plot(np.linspace(0, 0.5, (len(log_periodogram_y))), Y_spectrum_smooth, label="Y-log-periodogram", alpha=0.25)
plt.legend()
plt.savefig("opt_securities_power_spectrum_Y")
plt.show()

# Plot element-wise difference - is there any trend here?
plt.plot(X_spectrum_smooth - Y_spectrum_smooth, label="Element-wise-first-difference")
plt.plot(savgol_filter((X_spectrum_smooth - Y_spectrum_smooth), 15001, 2), label="Smoothed-difference-in-XY-spectra")
plt.savefig("Opt_securities_power_spectrum_difference")
plt.show()

# Although there is not an obvious difference in periodic behaviour, the L1 difference plot indicates that there may be some trend in the differences
# Of course, we have to be aware of excessive smoothing
# It's possible that we see some medium-term reversal behaviour

# There is no notable difference in the Log power spectrum of the securities. Same peaks (implies limited intra-day, periodic phenomenon)
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

# # Split into Training and Testing set
# df_train = df.iloc[:50000,:]
# df_test = df.iloc[50000:,:]

# Write a function to generate a table over the space which may identify dislocations
def dislocation_function(df, grid_periodicity):
    # Loop over all grid components and assess divergence in correlation
    grid = np.linspace(np.min(df[grid_periodicity]), np.max(df[grid_periodicity]), len(np.unique(df[grid_periodicity])))
    metrics_periodicity_list = []
    for i in range(len(grid)):
        slice = df.loc[df[grid_periodicity] == grid[i]]
        bid_correlation = slice["X_BID_log_returns"].corr(slice["Y_BID_log_returns"])
        ask_correlation = slice["X_ASK_log_returns"].corr(slice["Y_ASK_log_returns"])
        x_bid_liquidity = slice["X_BID_VOL"].mean()
        x_ask_liquidity = slice["X_ASK_VOL"].mean()
        y_bid_liquidity = slice["X_BID_VOL"].mean()
        y_ask_liquidity = slice["X_ASK_VOL"].mean()

        X_spread = slice["X_Spread"].mean()
        Y_spread = slice["Y_Spread"].mean()
        x_returns_mean = slice["X_BID_log_returns"].mean()
        x_returns_sd = slice["X_BID_log_returns"].std()
        y_returns_mean = slice["Y_BID_log_returns"].mean()
        y_returns_sd = slice["Y_BID_log_returns"].std()
        returns_mean_diff = x_returns_mean - y_returns_mean
        metrics_periodicity_list.append([grid[i], bid_correlation, ask_correlation, x_bid_liquidity, x_ask_liquidity,
                                         y_bid_liquidity, y_ask_liquidity, X_spread, Y_spread, x_returns_mean,
                                         y_returns_mean, x_returns_sd, y_returns_sd, returns_mean_diff])

    # Correlation scores minute array
    metrics_periodicity_df = pd.DataFrame(metrics_periodicity_list)
    metrics_periodicity_df.columns = ["Periodicity", "Bid_correlation", "Ask_correlation", "X_Bid_liquidity", "X_Ask_liquidity",
                                      "Y_Bid_liquidity", "Y_Ask_liquidity", "X_Spread", "Y_Spread", "X_log_returns",
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

### MODEL TRAINING ###

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
        x_spread = unique_minute_hour["X_Spread"].mean()
        y_spread = unique_minute_hour["Y_Spread"].mean()
        x_avg_returns = unique_minute_hour["X_BID_log_returns"].mean()
        y_avg_returns = unique_minute_hour["Y_BID_log_returns"].mean()
        x_y_avg_mispricing = x_avg_returns - y_avg_returns
        x_bid_liquidity = unique_minute_hour["X_BID_VOL"].mean()
        x_ask_liquidity = unique_minute_hour["X_ASK_VOL"].mean()
        y_bid_liquidity = unique_minute_hour["Y_BID_VOL"].mean()
        y_ask_liquidity = unique_minute_hour["Y_ASK_VOL"].mean()
        minute_hour_returns_list.append([hour_spacing[i], minute_spacing[j], len(unique_minute_hour), correlation_bid_xy,
                                         x_spread, y_spread, x_avg_returns, y_avg_returns,x_y_avg_mispricing,
                                         x_bid_liquidity, x_ask_liquidity, y_bid_liquidity, y_ask_liquidity])
# Convert to Dataframe
minute_hour_df = pd.DataFrame(minute_hour_returns_list)
minute_hour_df.columns = ["Hour", "Minute", "Samples", "Correlation", "X_avg_spread", "Y_avg_spread", "X_avg_returns", "Y_avg_returns", "X_Y_avg_mispricing",
                          "x_bid_liquidity", "x_ask_liquidity", "y_bid_liquidity", "y_ask_liquidity"]

# On average, throughout the day - how do these stocks move?
x_bid = df["X_BID"]
y_bid = df["Y_BID"]
x_cleaned_bid = x_bid.fillna(method='bfill')
y_cleaned_bid = y_bid.fillna(method='bfill')
x_returns = pd.DataFrame(minute_hour_df["X_avg_returns"])
y_returns = pd.DataFrame(minute_hour_df["Y_avg_returns"])
x_cleaned_returns = x_returns.fillna(0)
y_cleaned_returns = y_returns.fillna(0)

# Plot Ratio of one time series to the other (this is meant to be stationary)
plt.plot(np.nan_to_num(df["X_BID"]/df["Y_BID"]))
plt.show()

# Liquidity Throughout the day (bid-ask spread)
plt.plot(minute_hour_df["X_avg_spread"], label="X-liquidity-spread")
plt.plot(minute_hour_df["Y_avg_spread"], label="Y-liquidity-spread")
plt.savefig("Opt_Liquidity_over_time")
plt.show()

# Liquidity Throughout the day (bid-ask spread)
plt.plot(minute_hour_df["x_bid_liquidity"], label = "X-liquidity-volume-bid")
plt.plot(minute_hour_df["x_ask_liquidity"], label = "X-liquidity-volume-ask")
plt.plot(minute_hour_df["y_bid_liquidity"], label = "Y-liquidity-volume-bid")
plt.plot(minute_hour_df["y_ask_liquidity"], label = "Y-liquidity-volume-ask")
plt.legend()
plt.savefig("Opt_liquidity_volume")
plt.show()

# Spot Divergence throughout the average day (subject to liquidity)


# # Test for correlation and cointegration
# print('Correlation: ' + str(np.corrcoef(np.array(df["X_BID"]), np.array(df["Y_BID"]))))
# score, pvalue, _ = coint(x_cleaned_bid, y_cleaned_bid)
# print('Cointegration test p-value: ' + str(pvalue))
#
# # Based on cointegration score between cleaned time series - we cannot reject null

# Moving average function
def moving_average(returns, window):
    moving_avg = []
    for i in range(window, len(returns)):
        moving_avg.append(np.mean(returns[i - window:i]))
    return moving_avg

# The series appear to be both correlated and cointegrated
# Compute Cumulative returns from X & Y
x_avg_cum_returns = x_cleaned_returns.cumsum()
y_avg_cum_returns = y_cleaned_returns.cumsum()

# Smooth out returns with X & Y (20 minute blocks)
x_avg_cum_returns_smoothed = moving_average(x_avg_cum_returns, 10)
y_avg_cum_returns_smoothed = moving_average(y_avg_cum_returns, 10)
difference_smoothed_ts = np.array(x_avg_cum_returns_smoothed) - np.array(y_avg_cum_returns_smoothed)

if make_plots:
    # Plot of cumulative log returns for X vs Y
    plt.plot(x_avg_cum_returns, label='X-avg time series')
    plt.plot(y_avg_cum_returns, label='Y-avg time series')
    plt.plot(x_avg_cum_returns_smoothed, label='X-avg time series smoothed')
    plt.plot(y_avg_cum_returns_smoothed, label='Y-avg time series smoothed')
    plt.xlabel("Minutes throughout the day")
    plt.ylabel("Avg cumulative returns")
    plt.legend()
    plt.savefig("Opt_avg_cumulative_returns")
    plt.show()

# Plot the difference in smoothed Time series vs Time
plt.plot(np.linspace(0,len(difference_smoothed_ts),len(difference_smoothed_ts)), difference_smoothed_ts)
plt.xlabel("Minutes in day")
plt.ylabel("Difference in smoothed TS")
plt.title("Evolutionary_difference")
plt.savefig("Opt_Evolutionary_difference_direction")
plt.show()

# Plot the difference in smoothed Time series vs Time
plt.plot(np.linspace(0,len(difference_smoothed_ts),len(difference_smoothed_ts)), np.abs(difference_smoothed_ts))
plt.xlabel("Minutes in day")
plt.ylabel("Difference in smoothed TS")
plt.title("Evolutionary_difference")
plt.savefig("Opt_Evolutionary_difference_modulus")
plt.show()

# Identify locations of buy/sell potential
argmax_100 = np.argmax(difference_smoothed_ts[0:100])
argmin_100 = np.argmin(difference_smoothed_ts[0:100])
argmax_200_400 = 200 + np.argmax(difference_smoothed_ts[200:])

# df = pd.read_csv('final_data_10s.csv', index_col='Time',parse_dates=True)
# dat_10am=df.loc[(df["Hour"]==8) & (df["Minute"]==11) & (df["X_Spread"]<20) & (df["Y_Spread"]<20)]
# dat_11am=df.loc[(df["Hour"]==13) & (df["Minute"]==55)]
# dat_11am.index=dat_11am.index.date
# strat1=dat_11am.X_BID+dat_10am.Y_BID - dat_10am.X_ASK -dat_11am.Y_ASK #long X, short Y
# strat1.describe()
