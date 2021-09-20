import numpy as np
import matplotlib.pyplot as plt
from Utilities import dendrogram_plot, dendrogram_plot_test
import pandas as pd
import glob

# plot parameters
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

labels = ["2,2","2,3","2,4","2,5","2,6","2,7","2,8","2,9",
          "3,2","3,3","3,4","3,5","3,6","3,7","3,8","3,9",
          "4,2","4,3","4,4","4,5","4,6","4,7","4,8","4,9",
          "5,2","5,3","5,4","5,5","5,6","5,7","5,8","5,9",
          "6,2","6,3","6,4","6,5","6,6","6,7","6,8","6,9",
          "7,2","7,3","7,4","7,5","7,6","7,7","7,8","7,9",
          "8,2","8,3","8,4","8,5","8,6","8,7","8,8","8,9",
          "9,2","9,3","9,4","9,5","9,6","9,7","9,8","9,9"]

# Read in data
path = '/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1' # use your path
all_files = glob.glob(path + "/*.csv")
all_files.sort()

lamda_1_list = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df_slice = np.array(df.iloc[1,1:])
    lamda_1_list.append(df_slice)

# Convert to an array
distance_matrix = np.zeros((len(lamda_1_list),len(lamda_1_list)))
for i in range(len(lamda_1_list)):
    for j in range(len(lamda_1_list)):
        lamda_1_i = lamda_1_list[i]
        lamda_1_j = lamda_1_list[j]

        # Compute L1 distance between vectors
        dist = np.sum(np.abs(lamda_1_i - lamda_1_j))
        distance_matrix[i,j] = dist
    print("Iteration", i)

# Plot heatmap
fig, ax = plt.subplots()
im = ax.imshow(distance_matrix)
# We want to show all ticks...
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.savefig("lambda_1_heatmap")
plt.show()

# Dendrogram plot labels
dendrogram_plot_test(distance_matrix, "_L1_", "lambda_1_", labels)