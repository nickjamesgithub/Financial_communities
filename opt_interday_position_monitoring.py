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
from statsmodels.tsa.stattools import coint, adfuller
from scipy.signal import savgol_filter
from scipy.stats import wasserstein_distance
import statsmodels.api as sm
import datetime

make_plots = True

# Read in the data
df = pd.read_csv("/Users/tassjames/Desktop/optiver_assessment/final_data_10s.csv")
df["Date"] = df["Time"].apply(lambda x: datetime.datetime.fromisoformat(x).date())

# Compute Log returns of X & Y
df['X_BID_log_returns'] = np.log(df["X_BID"]) - np.log(df["X_BID"].shift(1))
df['Y_BID_log_returns'] = np.log(df["Y_BID"]) - np.log(df["Y_BID"].shift(1))
df['X_ASK_log_returns'] = np.log(df["X_ASK"]) - np.log(df["X_ASK"].shift(1))
df['Y_ASK_log_returns'] = np.log(df["Y_ASK"]) - np.log(df["Y_ASK"].shift(1))

# Group Data by date
df_grouped = df.groupby("Date").mean()

# Plot Ratio of X-BID:Y-BID
ratio_xy = df_grouped["X_BID"]/df_grouped["Y_BID"]
plt.plot(ratio_xy)
plt.show()

def zscore(series):
    return (series - series.mean()) / np.std(series)

# Compute Z scores
z_scores = zscore(ratio_xy)

S1 = df_grouped["X_BID"]
S2 = df_grouped["Y_BID"]
ratios = S1 / S2
z_scores = z_scores

# Generate unique date grid
date_grid_unique = df["Date"].unique()

# Simulate trading
curr_money = 0
money_list = [0]
m2m_list = []
iterations = len(ratios)
state = "neutral" # neutral, long_X, short_X
state_list = ["neutral", "neutral"] # Cold start
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
        if z_scores[i] < -1:  # go long on X and short on Y
            curr_money += df["Y_BID"][i] - df["X_ASK"][i]
            state = "long_X"
            long_x_decision.append(i)
            long_counter += 1
            state_list.append(state)
        if z_scores[i] > 1:  # go long on Y and short on X
            curr_money += df["X_BID"][i] - df["Y_ASK"][i]
            state = "short_X"
            short_x_decision.append(i)
            short_counter += 1
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
        if z_scores[i] > 0:  # close out position, ie sell X, buy Y
            curr_money += df["X_BID"][i] - df["Y_ASK"][i]
            state = "neutral"
            exit_long_x_decision.append(i)
            long_counter -= 1
        money_list.append(curr_money)
        long_counter_list.append(long_counter)
        short_counter_list.append(short_counter)
        state_list.append(state)
    if state == "short_X":
        if z_scores[i] < 0:  # Close out position, ie sell Y, buy X
            curr_money += df["Y_BID"][i] - df["X_ASK"][i]
            state = "neutral"
            exit_short_x_decision.append(i)
            short_counter -= 1
        money_list.append(curr_money)
        long_counter_list.append(long_counter)
        short_counter_list.append(short_counter)
        state_list.append(state)

# Fix state list for cold start problem
state_list_cs_fix = state_list[2:]
money_list_cs_fix = money_list[0:]

# Generate grid for plots
grid = np.linspace(1,len(ratio_xy),len(ratio_xy))

# Long and Short Strategy Counter List
plt.plot(grid, long_counter_list, label="Long_X_strategy")
plt.plot(grid, short_counter_list, label="Short_X_strategy")
plt.plot(grid, zscore(ratio_xy), color='black')
plt.legend()
plt.savefig("Opt_Long_Short_strategy_counter_list")
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