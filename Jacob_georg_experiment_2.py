import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Choose number of sectors and n for simulation
sectors_list = [2,3,4,5,6,7,8,9] # 2,3,4,5,6,7,8,9
samples_list = [2,3,4,5,6,7,8,9] # 2,3,4,5,6,7,8,9
crisis_list = ["dot_com", "gfc", "covid", "ukraine"] # "dot_com", "gfc", "covid", "ukraine"

# Loop over crises, sectors and samples
crisis_ = []
lamda_mean_crisis = []
sector_stock_crisis = []
for c in range(len(crisis_list)):
    lamda_mean_list = []
    sector_stock_list = []
    for k in range(len(sectors_list)):
        for s in range(len(samples_list)):
            data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/lambda_paths/"+"_"+crisis_list[c]+"_"
                               +str(sectors_list[k])+'_'+str(samples_list[s])+".csv")
            lamda_mean = np.mean(data.iloc[1,1:])
            lamda_mean_list.append(lamda_mean)
            sector_stock_list.append([sectors_list[k], samples_list[s]])
            print("Crisis: ", crisis_list[c])
            print("# Sectors:  ", sectors_list[k])
            print("# Stocks:  ", samples_list[s])
    # Append to global store lists
    crisis_.append(crisis_list[c])
    lamda_mean_crisis.append(lamda_mean_list)
    sector_stock_crisis.append(sector_stock_list)

    # Convert to Dataframe and write to csv
    lamda_mean_df = pd.DataFrame(lamda_mean_list)
    lamda_mean_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/lambda_path_outputs/" + "_" + crisis_list[c] + "_" + ".csv")