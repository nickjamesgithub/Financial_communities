import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
import datetime

make_plots = False

# Set path to read in lambda 1 data
path = '/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1/' # use your path
all_files = glob.glob(path + "/*.csv")

variance_list = []

# Loop over filenames
for filename in all_files:

    # Get filename
    file_id = filename.rsplit('/', 1)[-1]
    df = pd.read_csv(filename)
    l_5_slice = np.array(df.iloc[0,1:])
    l_95_slice = np.array(df.iloc[2, 1:])
    variance = np.mean(l_95_slice - l_5_slice)
    variance_list.append([file_id, variance])
    print(filename)

variance_list_df = pd.DataFrame(variance_list)
variance_list_df.to_csv("/Users/tassjames/Desktop/Variance_list.csv")

