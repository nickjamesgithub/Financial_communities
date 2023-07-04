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
make_plots = True

# Read in the data
df = pd.read_csv("/Users/tassjames/Desktop/optiver_assessment/final_data_10s.csv")

# Convert Time column to Date Time object
df["Time"] = pd.to_datetime(df["Time"])
df["X_Spread"] = df["X_BID"] - df["X_ASK"]
df["Y_Spread"] = df["Y_BID"] - df["Y_ASK"]

# Plot Ratio of X-BID:Y-BID
ratio_xy = df["X_BID"]/df["Y_BID"]
plt.plot(ratio_xy)
plt.show()

def zscore(series):
    return (series - series.mean()) / np.std(series)

# Plot Z Score on top of Ratio
zscore(ratio_xy).plot(figsize=(12,6))
plt.axhline(zscore(ratio_xy).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.show()


# # Trade using a simple strategy
# def trade(df, zscore):
#     S1 = df["X_BID"]
#     S2 = df["Y_BID"]
#     ratios = S1/S2
#
#     # Simulate trading
#     # Start with no money and no positions
#     money = 0
#     countS1 = 0
#     countS2 = 0
#     cum_returns = []
#     for i in range(len(ratios)):
#         print("Ratio is", ratios[i])
#         print("Z-score is", z_scores[i])
#         # Sell short if the z-score is > 1
#         if zscore[i] < -1:
#             money += df["X_ASK"][i] - df["Y_BID"][i] * ratios[i]
#             # money += S1[i] - S2[i] * ratios[i]
#             countS1 -= 1
#             countS2 += ratios[i]
#             # print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
#         # Buy long if the z-score is < -1
#         elif zscore[i] > 1:
#             money -= df["X_BID"] - df["Y_ASK"][i] * ratios[i]
#             # money -= S1[i] - S2[i] * ratios[i]
#             countS1 += 1
#             countS2 -= ratios[i]
#             # print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
#         # Clear positions if the z-score between -.5 and .5
#         elif abs(zscore[i]) < 0.5:
#             #todo FIX THIS
#             money += S1[i] * countS1 + S2[i] * countS2
#             countS1 = 0
#             countS2 = 0
#             # print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
#         print("Iteration ", i)
#
#     print(money)
#     return money

z_scores = zscore(ratio_xy)
# trade(df, z_scores)

S1 = df["X_BID"]
S2 = df["Y_BID"]
ratios = S1 / S2
z_scores = z_scores

# Simulate trading
curr_money = 0
money_list = [0]
iterations = len(ratios)
state = "NEUTRAL"
state_list = []
z_scores_long_x = []
z_scores_short_x = []
long_x_decision = []
short_x_decision = []
exit_long_x_decision = []
exit_short_x_decision = []
for i in range(iterations):
    print(state)
    print("Iteration", i)
    # print("Ratio is", ratios[i])
    print("Z-score is", z_scores[i])
    # Sell short if the z-score is > 1
    state_list.append(state)
    if state == "neutral":
        if z_scores[i] < -1:  # go long on X and short on Y
            curr_money += df["Y_BID"][i] - df["X_ASK"][i]
            state = "long_X"
            long_x_decision.append(i)
        if z_scores[i] > 1:  # go long on Y and short on X
            curr_money += df["X_BID"][i] - df["Y_ASK"][i]
            state = "long_Y"
            short_x_decision.append(i)
        money_list.append(curr_money)
        state_list.append(state)
    if state == "long_X":
        if z_scores[i] > 0:  # close out position, ie sell X, buy Y
            curr_money += df["X_BID"][i] - df["Y_ASK"][i]
            state = "neutral"
            exit_long_x_decision.append(i)
        money_list.append(curr_money)
        state_list.append(state)
    if state == "long_Y":
        if z_scores[i] < 0: # Close out position, ie sell Y, buy X
            curr_money += df["Y_BID"][i] - df["X_ASK"][i]
            state = "neutral"
            exit_short_x_decision.append(i)
        money_list.append(curr_money)
        state_list.append(state)

    # # this is all within a big for loop on i
    # if state == "neutral":
    #     if Z_score < -1:  # go long on X and short on Y
    #         curr_money += Y_BID[i] - X_ASK[i]
    #         state = "long_X"
    #     if Z_score > 1:  # go long on Y and short on X
    #         curr_money += X_BID[i] - Y_ASK[i]
    #         state = "long_Y"
    #     money_list.append(curr_money)
    # if state == "long_X":
    #     if Z_score > 0:  # close out position, ie sell X, buy Y
    #         curr_money += X_BID[i] - Y_ASK[i]
    #         state = "neutral"
    #     money_list.append(curr_money)
    # if state == "long_Y":
    #     if Z_score < 0:
    #         curr_money += Y_BID[i] - X_ASK[i]
    #         state = "neutral"
    #     money_list.append(curr_money)

# Print Evolutionary money
grid = np.linspace(1,len(money_list),len(money_list))
# plt.plot(grid, money_list)
plt.plot(grid, zscore(ratio_xy), color='black')
plt.axhline(zscore(ratio_xy).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
for i in range(len(exit_short_x_decision)):
    plt.axvspan(int(grid[short_x_decision[i][0]]), int(grid[exit_short_x_decision[i][0]]), color="red", label="Short_X",alpha=0.1)
for j in range(len(exit_long_x_decision)):
    plt.axvspan(int(grid[long_x_decision[j][0]]), int(grid[exit_long_x_decision[j][0]]), color="green", label="Long_X", alpha=0.1)
plt.savefig("Opt_Trade_positions")
plt.show()

x=1
y=2