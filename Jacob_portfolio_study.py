import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crisis = "ukraine" # covid, dot_com, gfc, ukraine

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
    print(ordered_merge_sectors)

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
    print(ordered_merge_sectors)

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
    print(ordered_merge_sectors)

# UKRAINE
if crisis == "ukraine":
    data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/results/portfolio_simulation_results_ukraine.csv")
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
    print(ordered_merge_sectors)