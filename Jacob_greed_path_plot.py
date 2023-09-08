import numpy as np
import matplotlib.pyplot as plt

# Crises
dot_com = [.435, .341, .311, .289, .268, .255, .248, .237, .232, .228, .221]
gfc = [.581, .527, .501, .479, .470, .465, .451, .452, .444, .442, .439, .439]
covid = [.631, .578, .555, .533, .516, .501, .514, .507, .507, .497, .502, .500]
ukraine = [0.512, 0.457, 0.414, 0.402, 0.37, 0.37, 0.365, 0.364, 0.351, 0.353, 0.351]

# Plot of 4 line plots
grid_dc = np.linspace(1,len(dot_com),len(dot_com))
grid_gfc = np.linspace(1,len(gfc),len(gfc))
grid_covid = np.linspace(1,len(covid),len(covid))
grid_ukraine = np.linspace(1,len(ukraine),len(ukraine))

# plot greedy paths
plt.scatter(grid_dc, dot_com, label= "Dot-com", marker='x')
plt.scatter(grid_gfc, gfc, label= "GFC", marker='x')
plt.scatter(grid_covid, covid, label= "Covid-19", marker='x')
plt.scatter(grid_ukraine, ukraine, label= "Ukraine", marker='x')
plt.xlabel("Number of diversification decisions")
plt.ylabel("Average collectivity")
plt.legend()
plt.savefig("Greedy_paths")
plt.show()