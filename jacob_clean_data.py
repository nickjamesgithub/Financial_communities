import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in data
data = pd.read_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors.csv")
data_slice = data.iloc[1:,1:]
data_clean = data.dropna(axis=1)

# Clean data
data_clean.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/Jacob_data_sectors_clean.csv")