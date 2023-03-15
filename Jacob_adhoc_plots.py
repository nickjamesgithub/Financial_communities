import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SECTOR SEQUENTIAL DIVERSIFICATION
dot_com_sector = [0.348, 0.294, 0.269, 0.253, 0.242, 0.236, 0.23, 0.225]
gfc_sector = [0.525, 0.488, 0.47, 0.461, 0.454, 0.448, 0.445, 0.442]
covid_sector = [0.577, 0.549, 0.529, 0.515, 0.511, 0.507, 0.504, 0.503]
ukraine_sector = [0.471, 0.417, 0.394, 0.374, 0.367, 0.365, 0.356, 0.356]

# STOCK SEQUENTIAL DIVERSIFICATION
dot_com_stock = [0.300,0.278, 0.265, 0.259, 0.254, 0.251, 0.245, 0.245]
gfc_stock = [0.493, 0.478, 0.468, 0.465, 0.46, 0.457, 0.456, 0.455]
covid_stock = [0.547, 0.531, 0.528, 0.52, 0.52, 0.52, 0.515, 0.514]
ukraine_stock = [0.419, 0.392, 0.387, 0.386, 0.386, 0.38, 0.377, 0.375]

# Plot across stock diversification
grid = np.linspace(2,9,len(dot_com_stock))
plt.plot(grid, dot_com_stock, label="Dot-com stock", color='red')
plt.plot(grid, dot_com_sector, label="Dot-com sector", color='red', linestyle='dotted')
plt.plot(grid, gfc_stock, label="GFC stock", color='blue')
plt.plot(grid, gfc_sector, label="GFC sector", color='blue', linestyle='dotted')
plt.plot(grid, covid_stock, label="COVID-19 stock", color='green')
plt.plot(grid, covid_sector, label="COVID-19 sector", color='green', linestyle='dotted')
plt.plot(grid, ukraine_stock, label="Ukraine stock", color='black', alpha=0.3)
plt.plot(grid, ukraine_sector, label="Ukraine sector", color='black', alpha=0.3, linestyle='dotted')
plt.xlabel("Number of Stocks")
plt.ylabel("Average market collectivity")
plt.legend()
plt.savefig("Sequential_diversification")
plt.show()

# Covid paths
covid_4_9 = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/lambda_paths/_covid_4_9.csv")
covid_9_4 = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/lambda_paths/_covid_9_4.csv")


def get_lambda_terms(df):
    l_95 = df.iloc[0,1:]
    l_50 = df.iloc[1, 1:]
    l_5 = df.iloc[2, 1:]
    return l_95, l_50, l_5

# Get uncertainty bounds and lambda_1 for all portfolio combinations
l_95_49, l_50_49, l_5_49 = get_lambda_terms(covid_4_9)
l_95_94, l_50_94, l_5_94 = get_lambda_terms(covid_9_4)
date_index_plot_covid = pd.date_range('13-03-2020','02-09-2020',len(l_95_49)).strftime('%Y-%m-%d')
# dot com - 3/3/2000 - 3/3/2002
# gfc - 5/1/2007 - 5/5/2010
# plot lambda 1 and the uncertainties
fig, ax = plt.subplots()
plt.plot(date_index_plot_covid, l_50_49, color='red', label="4,9 mean") # 3,9
plt.fill_between(date_index_plot_covid, l_5_49, l_95_49, alpha=0.1, color='red', label="4,9 uncertainty") # 3,9
plt.plot(date_index_plot_covid, l_50_94, color='blue', label="9,4 mean") # 9,3
plt.fill_between(date_index_plot_covid, l_5_94, l_95_94, alpha=0.1, color='blue', label="9,4 uncertainty") # 9,3
plt.tick_params(axis='x', which='major', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.ylabel(r'$\lambda_1$ / $N$')
plt.savefig("Lambda_1_variance_covid_94_49")
plt.show()