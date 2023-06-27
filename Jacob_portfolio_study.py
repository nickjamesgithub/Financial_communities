import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crisis = "crisis_distance" # covid, dot_com, gfc, ukraine, crisis_distance

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_real.csv", index_col='Date')
prices.columns = prices.columns.str.replace('(\.\d+)$', '')
v_counts = prices.columns.value_counts()
v_counts_sum = np.sum(v_counts)

# Overall percentage
v_counts_percentage = pd.DataFrame(v_counts / v_counts_sum)
v_counts_percentage.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/index_sampling_static.csv")

# COVID
if crisis == "covid":
    # Import data and slice top equities
    data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/portfolio_simulation_results_covid.csv")
    data_sort = data.sort_values('0')
    top_portfolios = data_sort.iloc[99000:100000,:]

    # Slice for top portfolios by equities and sectors
    top_portfolios_equities = top_portfolios.iloc[:,2:42]
    top_portfolios_sectors = top_portfolios.iloc[:, 42:82]

    # Convert to numpy vector and flatten
    top_equities_flat = top_portfolios_equities.values.reshape(-1)
    top_sectors_flat = top_portfolios_sectors.values.reshape(-1)

    # Flattened Equity merge
    equities_unique, equities_counts = np.unique(top_equities_flat, return_counts=True)
    merge_equities = np.stack((equities_unique, equities_counts), axis=1)
    ordered_merge_equities = merge_equities[np.argsort(merge_equities[:,1])]

    # Flattened Sectors merge
    sectors_unique, sectors_counts = np.unique(top_sectors_flat, return_counts=True)
    merge_sectors = np.stack((sectors_unique, sectors_counts), axis=1)
    ordered_merge_sectors = merge_sectors[np.argsort(merge_sectors[:,1])]

    # Convert to pandas dataframe
    pd_ordered_merge_sectors = pd.DataFrame(ordered_merge_sectors)
    sampling_sectors_count_sum = np.sum(pd_ordered_merge_sectors[1])

    # Overall percentage
    sampling_sector_names = pd_ordered_merge_sectors[0]
    sampling_sector_count_percentage = pd_ordered_merge_sectors[1] / sampling_sectors_count_sum
    covid_sampling_sector_percentages = pd.merge(sampling_sector_names, sampling_sector_count_percentage, right_index=True, left_index=True)
    covid_sampling_sector_percentages.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/covid_sampling_results.csv")

# Dot-com
if crisis == "dot_com":
    data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/portfolio_simulation_results_dot_com.csv")
    data_sort = data.sort_values('0')
    top_portfolios = data_sort.iloc[99000:100000,:]

    # Slice for top portfolios by equities and sectors
    top_portfolios_equities = top_portfolios.iloc[:,2:42]
    top_portfolios_sectors = top_portfolios.iloc[:, 42:82]

    # Convert to numpy vector and flatten
    top_equities_flat = top_portfolios_equities.values.reshape(-1)
    top_sectors_flat = top_portfolios_sectors.values.reshape(-1)

    # Flattened Equity merge
    equities_unique, equities_counts = np.unique(top_equities_flat, return_counts=True)
    merge_equities = np.stack((equities_unique, equities_counts), axis=1)
    ordered_merge_equities = merge_equities[np.argsort(merge_equities[:,1])]

    # Flattened Sectors merge
    sectors_unique, sectors_counts = np.unique(top_sectors_flat, return_counts=True)
    merge_sectors = np.stack((sectors_unique, sectors_counts), axis=1)
    ordered_merge_sectors = merge_sectors[np.argsort(merge_sectors[:,1])]

    # Convert to pandas dataframe
    pd_ordered_merge_sectors = pd.DataFrame(ordered_merge_sectors)
    sampling_sectors_count_sum = np.sum(pd_ordered_merge_sectors[1])

    # Overall percentage
    sampling_sector_names = pd_ordered_merge_sectors[0]
    sampling_sector_count_percentage = pd_ordered_merge_sectors[1] / sampling_sectors_count_sum
    dc_sampling_sector_percentages = pd.merge(sampling_sector_names, sampling_sector_count_percentage, right_index=True,
                                           left_index=True)
    dc_sampling_sector_percentages.to_csv(
        "/Users/tassjames/Desktop/jacob_financial_crises/results/dot_com_sampling_results.csv")

# GFC
if crisis == "gfc":
    data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/portfolio_simulation_results_gfc.csv")
    data_sort = data.sort_values('0')
    top_portfolios = data_sort.iloc[99000:100000,:]

    # Slice for top portfolios by equities and sectors
    top_portfolios_equities = top_portfolios.iloc[:,2:42]
    top_portfolios_sectors = top_portfolios.iloc[:, 42:82]

    # Convert to numpy vector and flatten
    top_equities_flat = top_portfolios_equities.values.reshape(-1)
    top_sectors_flat = top_portfolios_sectors.values.reshape(-1)

    # Flattened Equity merge
    equities_unique, equities_counts = np.unique(top_equities_flat, return_counts=True)
    merge_equities = np.stack((equities_unique, equities_counts), axis=1)
    ordered_merge_equities = merge_equities[np.argsort(merge_equities[:,1])]

    # Flattened Sectors merge
    sectors_unique, sectors_counts = np.unique(top_sectors_flat, return_counts=True)
    merge_sectors = np.stack((sectors_unique, sectors_counts), axis=1)
    ordered_merge_sectors = merge_sectors[np.argsort(merge_sectors[:,1])]

    # Convert to pandas dataframe
    pd_ordered_merge_sectors = pd.DataFrame(ordered_merge_sectors)
    sampling_sectors_count_sum = np.sum(pd_ordered_merge_sectors[1])

    # Overall percentage
    sampling_sector_names = pd_ordered_merge_sectors[0]
    sampling_sector_count_percentage = pd_ordered_merge_sectors[1] / sampling_sectors_count_sum
    gfc_sampling_sector_percentages = pd.merge(sampling_sector_names, sampling_sector_count_percentage, right_index=True, left_index=True)
    gfc_sampling_sector_percentages.to_csv(
        "/Users/tassjames/Desktop/jacob_financial_crises/results/gfc_sampling_results.csv")

# UKRAINE
if crisis == "ukraine":
    data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/portfolio_simulation_results_ukraine.csv")
    data_sort = data.sort_values('0')
    top_portfolios = data_sort.iloc[249000:250000,:]

    # Slice for top portfolios by equities and sectors
    top_portfolios_equities = top_portfolios.iloc[:,2:42]
    top_portfolios_sectors = top_portfolios.iloc[:, 42:82]

    # Convert to numpy vector and flatten
    top_equities_flat = top_portfolios_equities.values.reshape(-1)
    top_sectors_flat = top_portfolios_sectors.values.reshape(-1)

    # Flattened Equity merge
    equities_unique, equities_counts = np.unique(top_equities_flat, return_counts=True)
    merge_equities = np.stack((equities_unique, equities_counts), axis=1)
    ordered_merge_equities = merge_equities[np.argsort(merge_equities[:,1])]

    # Flattened Sectors merge
    sectors_unique, sectors_counts = np.unique(top_sectors_flat, return_counts=True)
    merge_sectors = np.stack((sectors_unique, sectors_counts), axis=1)
    ordered_merge_sectors = merge_sectors[np.argsort(merge_sectors[:,1])]

    # Convert to pandas dataframe
    pd_ordered_merge_sectors = pd.DataFrame(ordered_merge_sectors)
    sampling_sectors_count_sum = np.sum(pd_ordered_merge_sectors[1])

    # Overall percentage
    sampling_sector_names = pd_ordered_merge_sectors[0]
    sampling_sector_count_percentage = pd_ordered_merge_sectors[1] / sampling_sectors_count_sum
    ukraine_sampling_sector_percentages = pd.merge(sampling_sector_names, sampling_sector_count_percentage, right_index=True, left_index=True)

    ukraine_sampling_sector_percentages.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/ukraine_sampling_results.csv")

if crisis == "crisis_distance":
    data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Sampling_results_FC.csv")
    covid = (data["COVID"]).astype("float")
    dot_com = data["Dot-com"]
    gfc = data["GFC"]
    ukraine = data["Ukraine"]
    index = data["INDEX"]

    # Slice market periods
    periods = [dot_com, gfc, covid, ukraine, index]
    labels = ["Dot-com", "GFC", "Covid-19", "Ukraine", "Index"]

    dw_list = []
    for i in range(len(periods)):
        for j in range(len(periods)):
            discrete_wasserstein = np.sum(np.abs(periods[i]-periods[j]))/2
            dw_list.append(discrete_wasserstein)

    # Discrete wasserstein reshape
    dw_reshaped = np.array(dw_list).reshape(5,5)

    # Plot heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(dw_reshaped)
    fig.colorbar(cax)

    xaxis = np.arange(len(labels))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.savefig("Discrete_wasserstein_distance_matrix")
    plt.show()



