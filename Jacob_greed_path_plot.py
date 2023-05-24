import numpy as np
import matplotlib.pyplot as plt

# Crises
dot_com = [.435, .341, .311, .289, .268, .255, .248, .237, .232, .228, .221]
gfc = [.581, .527, .501, .479, .470, .465, .451, .452, .444, .442, .439, .439]
covid = [.631, .578, .555, .533, .516, .501, .514, .507, .507, .497, .502, .500]
ukraine = [0.512, 0.457, 0.414, 0.402, 0.37, 0.37, 0.365, 0.364, 0.351, 0.353, 0.351]

# Plot of 4 line plots
plt.plot(dot_com, label= "Dot-com")
plt.plot(gfc, label= "GFC")
plt.plot(covid, label= "Covid-19")
plt.plot(ukraine, label= "Ukraine")
plt.xlabel("# Securities in portfolio")
plt.ylabel("Average collectivity")
plt.savefig("Greedy_paths")
plt.legend()
plt.show()