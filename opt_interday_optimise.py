import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

make_plots = True
model = "Train" # "Train" or "Test" or "all"

# Read in the data
df = pd.read_csv(r"C:\Users\60848\Desktop\opt\final_data_10s.csv")
df["Date"] = df["Time"].apply(lambda x: datetime.datetime.fromisoformat(x).date())

# Split into Train and test data
training_length = int(0.75 * len(df))
testing_length = len(df) - training_length
if model == "Train":
    df = df.iloc[:training_length,:]
if model == "Test":
    df = df.iloc[-testing_length:,:]
if model == "all":
    df = df

# Compute Log returns of X & Y
df['X_BID_log_returns'] = np.log(df["X_BID"]) - np.log(df["X_BID"].shift(1))
df['Y_BID_log_returns'] = np.log(df["Y_BID"]) - np.log(df["Y_BID"].shift(1))
df['X_ASK_log_returns'] = np.log(df["X_ASK"]) - np.log(df["X_ASK"].shift(1))
df['Y_ASK_log_returns'] = np.log(df["Y_ASK"]) - np.log(df["Y_ASK"].shift(1))

# # Group Data by date
# df_grouped = df.groupby("Date").mean()

# Plot Ratio of X-BID:Y-BID
ratio_xy = df["X_BID"]/df["Y_BID"]
plt.plot(ratio_xy)
plt.show()

def zscore(series):
    return (series - series.mean()) / np.std(series)

# Compute Z scores
z_scores = zscore(ratio_xy)

S1 = df["X_BID"]
S2 = df["Y_BID"]
ratios = S1 / S2
z_scores = z_scores

# Generate unique date grid
date_grid_unique = df["Date"].unique()

# Simulate trading
def backtest(max_positions, long_x_parameter_open, long_x_parameter_close,
             short_x_parameter_open, short_x_parameter_close):
    curr_money = 0
    money_list = [0]
    iterations = len(ratios)
    state = "neutral"  # neutral, long_X, short_X
    state_list = ["neutral", "neutral"]  # Cold start
    z_scores_long_x = []
    z_scores_short_x = []
    long_x_decision = []
    short_x_decision = []
    exit_long_x_decision = []
    exit_short_x_decision = []
    strategy_profit = []
    long_counter = 0
    short_counter = 0
    long_counter_list = []
    short_counter_list = []

    for i in range(iterations):
        print(state)
        print("Iteration", i)
        print("Z-score is", z_scores[i])
    # Sell short if the z-score is > 1
        if state == "neutral":
            if z_scores[i] < long_x_parameter_open:  # go long on X and short on Y
                curr_money += df["X_BID"][i] - df["Y_ASK"][i]
                state = "short_X"
                short_x_decision.append(i)
                short_counter += max_positions
                state_list.append(state)
            if state_list[-2] == "neutral" and state_list[-1] != "neutral":
                print("transition")
                state_list.pop()
            else:
                money_list.append(curr_money)
                long_counter_list.append(long_counter)
                short_counter_list.append(short_counter)
                state_list.append(state)
        if state == "long_X":
            if z_scores[i] > long_x_parameter_close:  # close out position, ie sell X, buy Y
                curr_money += df["X_BID"][i] - df["Y_ASK"][i]
                state = "neutral"
                exit_long_x_decision.append(i)
                long_counter -= max_positions
            money_list.append(curr_money)
            long_counter_list.append(long_counter)
            short_counter_list.append(short_counter)
            state_list.append(state)
        if state == "short_X":
            if z_scores[i] < short_x_parameter_close:  # Close out position, ie sell Y, buy X
                curr_money += df["Y_BID"][i] - df["X_ASK"][i]
                state = "neutral"
                exit_short_x_decision.append(i)
                short_counter -= max_positions
            money_list.append(curr_money)
            long_counter_list.append(long_counter)
            short_counter_list.append(short_counter)
            state_list.append(state)

    # Fix state list for cold start problem
    state_list_cs_fix = state_list[2:]
    money_list_cs_fix = money_list[1:]

    return money_list_cs_fix, state_list_cs_fix, long_counter_list, short_counter_list, long_x_decision, exit_long_x_decision, short_x_decision, exit_short_x_decision

# Optimise model parameters
if model == "Train":
    long_x_open = [-1.5, -1.4, -1.3, -1.2, -1.1, -1, -.9, -.8, -.7]
    long_x_close = [-.5, -.4, -.3, -.2, -.1, 0]
    short_x_open = [1.5, 1.4, 1.3, 1.2, 1.1, 1, .9, .8, .7]
    short_x_close = [.5, .4, .3, .2, .1, 0]
if model == "Test":
    long_x_open = [-2]
    long_x_close = [-.1]
    short_x_open = [1.5]
    short_x_close = [.4]

results = []
for lo in range(len(long_x_open)):
    for lc in range(len(long_x_close)):
        for so in range(len(short_x_open)):
            for sc in range(len(short_x_close)):
                money_list_cs_fix, state_list_cs_fix, long_counter_list, short_counter_list, \
                long_x_decision, exit_long_x_decision, short_x_decision, exit_short_x_decision = backtest(max_positions=3, long_x_parameter_open=long_x_open[lo],
                                                                                                          long_x_parameter_close=long_x_close[lc],
                             short_x_parameter_open=short_x_open[so], short_x_parameter_close=short_x_close[sc])
                results.append([money_list_cs_fix[-1], long_x_open[lo],long_x_close[lc], short_x_open[so], short_x_close[sc]])
    print("iteration", long_x_open[lo])

# Make results list a dataframe
results_df = pd.DataFrame(results)
results_df.columns = ["Profit", "Long_open", "Long_close", "Short_open", "Short_close"]
results_df_sorted = results_df.sort_values(by="Profit")

# Generate grid for plots
grid = np.linspace(1,len(ratio_xy),len(ratio_xy))

# # Plot money over time
# plt.plot(grid, money_list_cs_fix)
# plt.xlabel("Days in strategy")
# plt.ylabel("Profit")
# plt.show()

# Long and Short Strategy Counter List
plt.plot(grid, long_counter_list, label="Long_X_strategy")
plt.plot(grid, short_counter_list, label="Short_X_strategy")
plt.plot(grid, zscore(ratio_xy), color='black')
plt.legend()
plt.savefig("Opt_Long_Short_strategy_counter_list_max_positions")
plt.show()

# Make list of states a Dataframe
state_df = pd.DataFrame(state_list_cs_fix)
state_df.columns = ["State"]
# Drop index for df_grouped
df_grouped = df_grouped.reset_index(drop=True)
# Merge State with
merge = pd.concat([df_grouped, state_df], axis=1)

# Print Evolutionary money
plt.plot(grid, zscore(ratio_xy), color='black')
plt.axhline(zscore(ratio_xy).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
for j in range(len(exit_short_x_decision)):
    plt.axvspan(int(grid[short_x_decision[j]]), int(grid[exit_short_x_decision[j]]), color="red", label="Short_X",alpha=0.1)
for k in range(len(exit_long_x_decision)):
    plt.axvspan(int(grid[long_x_decision[k]]), int(grid[exit_long_x_decision[k]]), color="green", label="Long_X", alpha=0.1)
plt.savefig("Opt_Trade_positions_daily_avg")
plt.show()