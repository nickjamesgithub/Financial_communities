import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

# Generate grid
x = np.linspace(0,10,1000)
y = np.sin(x)
noise_1 = np.random.normal(0,1,1000)
noise_5 = np.random.normal(0,5,1000)
y_noise_mid = y + noise_1
y_noise_high = y + noise_5

# Perturbation
plt.plot(x,y, label="y=sin(x)")
plt.scatter(x,y_noise_mid, label="y=sin(x) + small noise", alpha=0.15, color='red')
plt.scatter(x,y_noise_high, label="y=sin(x) + high noise", alpha=0.15, color='black')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.legend()
plt.savefig("Dummy_data")
plt.show()