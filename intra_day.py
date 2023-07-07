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
import itertools

make_plots = False
model = "All" # Train/Test/All
# Read in the data
df = pd.read_csv(r"C:\Users\60848\Desktop\opt\final_data_10s.csv")
if model == "Train":
    df = df.iloc[:400000, :]
if model == "Test":
    df = df.iloc[400000:,:]
if model == "All":
    df = df

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
    plt.savefig("Opt_Bid_TS_"+model)
    plt.show()

    # Plot the two time series
    plt.plot(df["Time"], df["X_ASK"])
    plt.plot(df["Time"], df["Y_ASK"])
    plt.xlabel("Time")
    plt.ylabel("Ask Price")
    plt.savefig("Opt_Ask_TS_"+model)
    plt.show()

    # Plot the two time series
    plt.plot(df["Time"], df["X_Spread"], label="X_liquidity")
    plt.plot(df["Time"], df["Y_Spread"], label="Y_liquidity")
    plt.xlabel("Time")
    plt.ylabel("Bid-Ask: X & Y")
    plt.savefig("Opt_XY_Liquidity_"+model)
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
window_size = (6 * 60 * 8) + 1 # Current smoothing corresponds to an 8 hour window (approximately a one-day trading window)
polynomial_order = 2
X_spectrum_smooth = np.nan_to_num(savgol_filter(log_periodogram_x, window_size, polynomial_order))
Y_spectrum_smooth = np.nan_to_num(savgol_filter(log_periodogram_y, window_size, polynomial_order))

if make_plots:
    # Plot power spectrum - X
    plt.plot(np.linspace(0, 0.5, (len(log_periodogram_x))), X_spectrum_smooth, label="X-log-periodogram", alpha=0.25)
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Log-PSD")
    plt.savefig("opt_securities_power_spectrum_X_"+model)
    plt.show()

    # Plot power spectrum - Y
    plt.plot(np.linspace(0, 0.5, (len(log_periodogram_y))), Y_spectrum_smooth, label="Y-log-periodogram", alpha=0.25)
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Log-PSD")
    plt.savefig("opt_securities_power_spectrum_Y_"+model)
    plt.show()

    # Plot element-wise difference - is there any trend here?
    plt.plot(np.linspace(0, 0.5, len(X_spectrum_smooth)), X_spectrum_smooth - Y_spectrum_smooth, label="Element-wise-first-difference")
    plt.plot(np.linspace(0, 0.5, len(X_spectrum_smooth)), savgol_filter((X_spectrum_smooth - Y_spectrum_smooth), 15001, 2), label="Smoothed-difference-in-XY-spectra")
    plt.xlabel("Frequency")
    plt.ylabel("L1 difference in Log-PSD")
    plt.savefig("Opt_securities_power_spectrum_difference_"+model)
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
df["Second"] = df["Time"].dt.second
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
second_spacing = np.arange(0,60,10)

# Grouped data by hour and minute
grouped_hour_minute = df.groupby(["Hour", "Minute"]).agg({k: np.mean for k in ['X_BID_log_returns', 'Y_BID_log_returns', 'X_Spread', 'Y_Spread']})

# Isolate first 10 seconds
unique_minute_hour_slice = df.loc[(df["Minute"]==0) & (df["Hour"]==8) & (df["Second"]==0)]
cum_sum_x = unique_minute_hour_slice["X_BID_log_returns"].fillna(0).cumsum()
cum_sum_y = unique_minute_hour_slice["Y_BID_log_returns"].fillna(0).cumsum()

# Plot cumulative returns of first 10 seconds
plt.plot(np.linspace(0,len(cum_sum_x),len(cum_sum_x)), cum_sum_x, label="Cumulative_returns_X")
plt.plot(np.linspace(0,len(cum_sum_y),len(cum_sum_y)), cum_sum_y, label = "Cumulative_returns_Y")
plt.xlabel("First 10 second instances")
plt.ylabel("Cumulative_returns")
plt.savefig("Cumulative_returns_X_Y_8_00_10")
plt.show()

# Loop over unique dates

up_start_days_x = []
up_start_days_y = []
down_start_days_x = []
down_start_days_y = []
for d in range(len(date_unique)):
    unique_date = df.loc[(df["Date"] == date_unique[d])]
    if unique_date["X_BID_log_returns"].iloc[1] > 0 and unique_date["Y_BID_log_returns"].iloc[1] > 0:
        up_start_days_x.append([list(unique_date["X_ASK_log_returns"].fillna(0))])
        up_start_days_y.append([list(unique_date["Y_ASK_log_returns"].fillna(0))])
    if unique_date["X_BID_log_returns"].iloc[1] < 0 and unique_date["Y_BID_log_returns"].iloc[1] < 0:
        down_start_days_x.append([list(unique_date["X_ASK_log_returns"].fillna(0))])
        down_start_days_y.append([list(unique_date["Y_ASK_log_returns"].fillna(0))])

# Flat data for histogram
up_start_x_array = np.array(list(itertools.chain.from_iterable(up_start_days_x)))
up_start_y_array = np.array(list(itertools.chain.from_iterable(up_start_days_y)))
down_start_x_array = np.array(list(itertools.chain.from_iterable(down_start_days_x)))
down_start_y_array = np.array(list(itertools.chain.from_iterable(down_start_days_y)))

# Determine consistency of argmin and argmax of elementwise difference curves
for i in range(len(up_start_x_array)):
    # Up difference curve
    up_start_x_cumsum = up_start_x_array[i,:].cumsum()
    up_start_y_cumsum = up_start_y_array[i,:].cumsum()

    up_xy_diffs = up_start_x_cumsum - up_start_y_cumsum
    argmin_up = np.argmin(up_xy_diffs)
    argmax_up = np.argmax(up_xy_diffs)

    plt.plot(up_start_x_cumsum)
    plt.plot(up_start_y_cumsum)
    plt.savefig("opt_xy_cumsum")
    plt.show()

    plt.plot(up_xy_diffs)
    plt.savefig("opt_xy_diffs")
    plt.show()

    # Down difference curve
    down_start_x_cumsum = down_start_x_array.cumsum()
    down_start_y_cumsum = down_start_y_array.cumsum()
    down_xy_diffs = down_start_x_cumsum - down_start_y_cumsum

#todo CLUSTER argmin and argmax of difference and see if there is a trend in the days

# Data in each minute/hour increment
minute_hour_returns_list = []
for i in range(len(hour_spacing)):
    for j in range(len(minute_spacing)):
        for k in range(len(second_spacing)):
            unique_minute_hour = df.loc[(df["Minute"]==minute_spacing[j]) & (df["Hour"]==hour_spacing[i]) & (df["Second"]==second_spacing[k])]
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
minute_hour_second_df = pd.DataFrame(minute_hour_returns_list)
minute_hour_second_df.columns = ["Hour", "Minute", "Samples", "Correlation", "X_avg_spread", "Y_avg_spread", "X_avg_returns", "Y_avg_returns", "X_Y_avg_mispricing",
                          "x_bid_liquidity", "x_ask_liquidity", "y_bid_liquidity", "y_ask_liquidity"]

# On average, throughout the day - how do these stocks move?
x_bid = df["X_BID"]
y_bid = df["Y_BID"]
x_cleaned_bid = x_bid.fillna(method='bfill')
y_cleaned_bid = y_bid.fillna(method='bfill')
x_returns = pd.DataFrame(minute_hour_second_df["X_avg_returns"])
y_returns = pd.DataFrame(minute_hour_second_df["Y_avg_returns"])
x_cleaned_returns = x_returns.fillna(0)
y_cleaned_returns = y_returns.fillna(0)

if make_plots:
    # Plot Ratio of one time series to the other (this is meant to be stationary)
    plt.plot(np.nan_to_num(df["X_BID"]/df["Y_BID"]))
    plt.ylabel("X/Y ratio")
    plt.xlabel("Time")
    plt.savefig("X_Y_Ratio_"+model)
    plt.show()

    # Liquidity Throughout the day (bid-ask spread)
    plt.plot(minute_hour_second_df["X_avg_spread"], label="X-liquidity-spread")
    plt.plot(minute_hour_second_df["Y_avg_spread"], label="Y-liquidity-spread")
    plt.xlabel("Expected_minute_per_day")
    plt.ylabel("Average_spread")
    plt.savefig("Opt_Liquidity_over_time_"+model)
    plt.show()

    # Liquidity Throughout the day (bid-ask spread)
    plt.plot(minute_hour_second_df["x_bid_liquidity"], label = "X-liquidity-volume-bid")
    plt.plot(minute_hour_second_df["x_ask_liquidity"], label = "X-liquidity-volume-ask")
    plt.plot(minute_hour_second_df["y_bid_liquidity"], label = "Y-liquidity-volume-bid")
    plt.plot(minute_hour_second_df["y_ask_liquidity"], label = "Y-liquidity-volume-ask")
    plt.xlabel("Expected_minute_per_day")
    plt.ylabel("Avg_volume")
    plt.legend()
    plt.savefig("Opt_liquidity_volume_"+model)
    plt.show()

# # Spot Divergence throughout the average day (subject to liquidity)
# # Test for correlation and cointegration between the two series
# print('Correlation: ' + str(np.corrcoef(np.array(df["X_BID"]), np.array(df["Y_BID"]))))
# score, pvalue, _ = coint(x_cleaned_bid, y_cleaned_bid)
# print('Cointegration test p-value: ' + str(pvalue))

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
x_avg_cum_returns_smoothed = moving_average(x_avg_cum_returns, 20)
y_avg_cum_returns_smoothed = moving_average(y_avg_cum_returns, 20)
difference_smoothed_ts = np.array(x_avg_cum_returns_smoothed) - np.array(y_avg_cum_returns_smoothed)

if make_plots:
    plt.plot(x_cleaned_returns, label="X_avg time series")
    plt.plot(y_cleaned_returns, label="Y_avg time series")
    plt.xlabel("Minutes throughout the day")
    plt.ylabel("Avg cumulative returns")
    plt.legend()
    plt.savefig("Opt_avg_returns_"+model)
    plt.show()

    # Plot of cumulative log returns for X vs Y
    plt.plot(x_avg_cum_returns, label='X-avg cumulative time series', alpha = 0.2)
    plt.plot(y_avg_cum_returns, label='Y-avg cumulative time series', alpha = 0.2)
    plt.plot(x_avg_cum_returns_smoothed, label='X-avg cumulative time series smoothed')
    plt.plot(y_avg_cum_returns_smoothed, label='Y-avg cumulative time series smoothed')
    plt.xlabel("Minutes throughout the day")
    plt.ylabel("Avg cumulative returns")
    plt.legend()
    plt.savefig("Opt_avg_cumulative_returns_"+model)
    plt.show()

    # Plot the difference in smoothed Time series vs Time
    plt.plot(np.linspace(0,len(difference_smoothed_ts),len(difference_smoothed_ts)), difference_smoothed_ts)
    plt.xlabel("Minutes in day")
    plt.ylabel("Difference in smoothed TS")
    plt.title("Evolutionary_difference")
    plt.savefig("Opt_Evolutionary_difference_direction_"+model)
    plt.show()

    # Plot the difference in smoothed Time series vs Time
    plt.plot(np.linspace(0,len(difference_smoothed_ts),len(difference_smoothed_ts)), np.abs(difference_smoothed_ts))
    plt.xlabel("Minutes in day")
    plt.ylabel("Difference in smoothed TS")
    plt.title("Evolutionary_difference")
    plt.savefig("Opt_Evolutionary_difference_modulus_"+model)
    plt.show()

# Identify locations of buy/sell potential
argmax_100 = np.argmax(difference_smoothed_ts[0:100])
argmin_100 = np.argmin(difference_smoothed_ts[0:100])
argmax_200_400 = 200 + np.argmax(difference_smoothed_ts[200:])

data = pd.read_csv(r"C:\Users\60848\Desktop\opt\final_data_10s.csv", index_col='Time', parse_dates=True)
data.index #see index is datetime format
data.loc['2020-08-01 08'] #Filter data based on one day, hour etc
price_names=['X_BID','X_ASK','Y_BID','Y_ASK']
vol_names=['X_BID_VOL','X_ASK_VOL','Y_BID_VOL','Y_ASK_VOL']
price_df=data[price_names];vol_df=data[vol_names]
data['X_spread']=data.X_ASK - data.X_BID; data['Y_spread']=data.Y_ASK - data.Y_BID
spread_df=data[['X_spread','Y_spread']]

# Intra-day trading strategy
# When we refer to LONG, we are referring to the ratio of X/Y
x_spread_ub = np.arange(10,50,10)
y_spread_ub = np.arange(10,50,10)
open_minute = np.linspace(5, 19, 15)
# open_second = np.arange(10, 60, 10)
close_minute = np.linspace(45, 59, 15)
# close_second = np.arange(10, 60, 10)

# We are going to take the 30 second point within each minute (however this could also be optimised)
intra_day_results = []
for xs in range(len(x_spread_ub)):
    for ys in range(len(y_spread_ub)):
        for om in range(len(open_minute)):
                for cm in range(len(close_minute)):

                    # Print open minute & close minute
                    print("Open minute ", int(open_minute[om]))
                    print("Close minute", int(close_minute[cm]))
                    print("X spread ", int(x_spread_ub[xs]))
                    print("Y spread", int(y_spread_ub[ys]))

                    # OPEN
                    open_time = datetime.time(8, np.int(open_minute[om]), np.int(30))
                    open_slice = data.loc[(data.index.time==open_time) & (spread_df.X_spread<x_spread_ub[xs]) & (spread_df.Y_spread<y_spread_ub[ys])]
                    open_slice.index=open_slice.index.date

                    # CLOSE
                    close_time = datetime.time(13, np.int(close_minute[cm]), np.int(30))
                    close_slice=data.loc[data.index.time==close_time]
                    close_slice.index=close_slice.index.date

                    #At Open (around 8:10), go long X and short Y, and close later in the day (around 155).
                    strategy = close_slice.X_BID + open_slice.Y_BID - open_slice.X_ASK - close_slice.Y_ASK
                    strategy_clean = strategy.fillna(0)
                    cumulative_profit = np.cumsum(strategy_clean)
                    intra_day_results.append([cumulative_profit[-1], open_minute[om], x_spread_ub[xs], close_minute[cm], y_spread_ub[ys]])

# Make intraday results a dataframe
intra_day_results_df = pd.DataFrame(intra_day_results)
intra_day_results_df.columns = ["Profit", "Open_minute", "X_spread", "Close_minute", "Y_spread"]
intra_day_results_df.sort_values(by="Profit")
x=1