import numpy as np
import matplotlib.pyplot as plt

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
plt.plot(grid, dot_com_stock, label="Dot-com stock")
plt.plot(grid, dot_com_sector, label="Dot-com sector")
plt.plot(grid, gfc_stock, label="GFC stock")
plt.plot(grid, gfc_sector, label="GFC sector")
plt.plot(grid, covid_stock, label="COVID-19 stock")
plt.plot(grid, covid_sector, label="COVID-19 sector")
plt.plot(grid, ukraine_stock, label="Ukraine stock")
plt.plot(grid, ukraine_sector, label="Ukraine sector")
plt.xlabel("Number of Stocks")
plt.ylabel("Average market collectivity")
plt.legend()
plt.savefig("Sequential_diversification")
plt.show()

# # Plot across sector diversification
# grid = np.linspace(2,9,len(dot_com_sector))
# plt.plot(grid, dot_com_sector, label="Dot-com sector")
# plt.plot(grid, gfc_sector, label="GFC sector")
# plt.plot(grid, covid_sector, label="COVID-19 sector")
# plt.plot(grid, ukraine_sector, label="Ukraine sector")
# plt.xlabel("Number of Sectors")
# plt.ylabel("Average market collectivity")
# plt.legend()
# plt.savefig("Sector_sequential_diversification")
# plt.show()