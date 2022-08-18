import numpy as np
import matplotlib.pyplot as plt
from Utilities import dendrogram_plot, dendrogram_plot_test
import pandas as pd
import glob

market_list = ["all"]

for i in range(len(market_list)):
    market_period_i = market_list[i]
    market_period = market_period_i # all, gfc, gfc_crash, interim, covid, covid_crash

    # plot parameters
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    labels = ["2,2","2,3","2,4","2,5","2,6","2,7","2,8","2,9",
              "3,2","3,3","3,4","3,5","3,6","3,7","3,8","3,9",
              "4,2","4,3","4,4","4,5","4,6","4,7","4,8","4,9",
              "5,2","5,3","5,4","5,5","5,6","5,7","5,8","5,9",
              "6,2","6,3","6,4","6,5","6,6","6,7","6,8","6,9",
              "7,2","7,3","7,4","7,5","7,6","7,7","7,8","7,9",
              "8,2","8,3","8,4","8,5","8,6","8,7","8,8","8,9",
              "9,2","9,3","9,4","9,5","9,6","9,7","9,8","9,9"] # Try 10 at the top

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
            if market_period == "all":
                lamda_1_i = lamda_1_list[i]
                lamda_1_j = lamda_1_list[j]
                # Compute L1 distance between vectors
                # dist = np.sum(np.abs(lamda_1_i - lamda_1_j))/len(lamda_1_i)
                dist = np.sum(np.abs(lamda_1_i - lamda_1_j))
                distance_matrix[i,j] = dist

            if market_period == "gfc":
                lamda_1_i = lamda_1_list[i][1828-120:2718-120]
                lamda_1_j = lamda_1_list[j][1828-120:2718-120]
                # Compute L1 distance between vectors
                dist = np.sum(np.abs(lamda_1_i - lamda_1_j))/len(lamda_1_i)
                distance_matrix[i,j] = dist

            if market_period == "gfc_crash":
                lamda_1_i = lamda_1_list[i][2264-120:2458-120]
                lamda_1_j = lamda_1_list[j][2264-120:2458-120]
                # Compute L1 distance between vectors
                dist = np.sum(np.abs(lamda_1_i - lamda_1_j))/len(lamda_1_i)
                distance_matrix[i,j] = dist

            if market_period == "interim":
                lamda_1_i = lamda_1_list[i][2719-120:5262-120]
                lamda_1_j = lamda_1_list[j][2719-120:5262-120]
                # Compute L1 distance between vectors
                dist = np.sum(np.abs(lamda_1_i - lamda_1_j))/len(lamda_1_i)
                distance_matrix[i,j] = dist
            if market_period == "covid_crash":
                lamda_1_i = lamda_1_list[i][5262-120:5326-120]
                lamda_1_j = lamda_1_list[j][5262-120:5326-120]
                # Compute L1 distance between vectors
                dist = np.sum(np.abs(lamda_1_i - lamda_1_j))/len(lamda_1_i)
                distance_matrix[i,j] = dist

            if market_period == "covid":
                lamda_1_i = lamda_1_list[i][5262-120:5392-120]
                lamda_1_j = lamda_1_list[j][5262-120:5392-120]
                # Compute L1 distance between vectors
                dist = np.sum(np.abs(lamda_1_i - lamda_1_j))/len(lamda_1_i)
                distance_matrix[i,j] = dist

        print("Iteration", i)

    # Dendrogram plot labels
    dendrogram_plot_test(distance_matrix, "_L1_", "lambda_1_L1"+market_period_i, labels)
    print(np.linalg.norm(distance_matrix))