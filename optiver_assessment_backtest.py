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


# Trade using a simple strategy
def trade(df, zscore):
    S1 = df["X_BID"]
    S2 = df["Y_BID"]
    ratios = S1/S2

    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    cum_returns = []
    for i in range(len(ratios)):
        print("Ratio is", ratios[i])
        print("Z-score is", z_scores[i])
        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += df["X_ASK"][i] - df["Y_BID"][i] * ratios[i]
            # money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            # print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= df["X_BID"] - df["Y_ASK"][i] * ratios[i]
            # money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            # print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.5:
            #todo FIX THIS
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            # print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
        print("Iteration ", i)

    print(money)
    return money

z_scores = zscore(ratio_xy)
# trade(df, z_scores)

S1 = df["X_BID"]
S2 = df["Y_BID"]
ratios = S1 / S2
z_scores = z_scores

# Simulate trading
# Start with no money and no positions
money = 0
iterations = 50000
flag = "NEUTRAL"
z_scores_long_x = []
z_scores_short_x = []
for i in range(iterations):
    print(flag)
    print("Iteration", i)
    # print("Ratio is", ratios[i])
    print("Z-score is", z_scores[i])
    # Sell short if the z-score is > 1
    if flag == "LONG_X":
        # returns = df["X_ASK"][i] - df["Y_BID"][i] * ratios[i]
        money += df["X_ASK"][i] - df["Y_BID"][i] * ratios[i]
        print(df["Time"][i])
        z_scores_long_x.append([z_scores[i], money, flag])
    if flag == "SHORT_X":
        money -= df["X_BID"] - df["Y_ASK"][i] * ratios[i]
        print(df["Time"][i])
        z_scores_short_x.append([z_scores[i], money, flag])
    if flag == "LONG_X" and z_scores[i] < -0.5:
        flag = "NEUTRAL"
    if flag == "SHORT_X" and z_scores[i] < 0.5:
        flag = "NEUTRAL"
    if flag == "NEUTRAL" and z_scores[i] < -1:
        flag = "LONG_X"
    if flag == "NEUTRAL" and z_scores[i] > 1:
        flag = "SHORT_X"
x=1
y=2
print(money)