import pandas as pd
import matplotlib.pyplot as plt

lambda_3_3 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1/lambda_paths_3_3.csv")
lambda_3_9 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1/lambda_paths_3_9.csv")
lambda_5_5 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1/lambda_paths_5_5.csv")
lambda_9_3 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1/lambda_paths_9_3.csv")
lambda_9_4 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1/lambda_paths_9_4.csv")
lambda_9_9 = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sampling_results/lambda_1/lambda_paths_9_9.csv")

def get_lambda_terms(df):
    l_95 = df.iloc[0,1:]
    l_50 = df.iloc[1, 1:]
    l_5 = df.iloc[2, 1:]
    return l_95, l_50, l_5

# Get uncertainty bounds and lambda_1 for all portfolio combinations
l_95_33, l_50_33, l_5_33 = get_lambda_terms(lambda_3_3)
l_95_39, l_50_39, l_5_39 = get_lambda_terms(lambda_3_3)
l_95_55, l_50_55, l_5_55 = get_lambda_terms(lambda_5_5)
l_95_93, l_50_93, l_5_93 = get_lambda_terms(lambda_9_3)
l_95_94, l_50_94, l_5_94 = get_lambda_terms(lambda_9_4)
l_95_99, l_50_99, l_5_99 = get_lambda_terms(lambda_9_9)
date_index_plot = pd.date_range('16-06-2000','08-10-2020',len(l_95_33)).strftime('%Y-%m-%d')

# plot lambda 1 and the uncertainties
fig, ax = plt.subplots()
plt.plot(date_index_plot, l_50_39, color='red', label="3,9 mean") # 3,9
plt.fill_between(date_index_plot, l_5_39, l_95_39, alpha=0.1, color='red', label="3,9 uncertainty") # 3,9
plt.plot(date_index_plot, l_50_93, color='blue', label="9,3 mean") # 9,3
plt.fill_between(date_index_plot, l_5_93, l_95_93, alpha=0.1, color='blue', label="9,3 uncertainty") # 9,3
plt.tick_params(axis='x', which='major', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.ylabel(r'$\lambda_1$ / $N$')
plt.savefig("Lambda_1_variance_93_39")
plt.show()

fig, ax = plt.subplots()
plt.plot(date_index_plot, l_50_94, color='blue', label="9,4 mean") # 9,4
plt.fill_between(date_index_plot, l_5_94, l_95_94, alpha=0.1, color='blue', label="9,4 uncertainty") # 9,4
plt.plot(date_index_plot, l_50_99, color='red', label="9,9 mean") # 9,9
plt.fill_between(date_index_plot, l_5_99, l_95_99, alpha=0.1, color='red', label="9,9 uncertainty") # 9,9
plt.tick_params(axis='x', which='major', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.ylabel(r'$\lambda_1$ / $N$')
plt.savefig("Lambda_1_variance_94_99")
plt.show()