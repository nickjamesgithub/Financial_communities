import matplotlib.pyplot as plt
import numpy as np

# Definition of the Marchenko-Pastur density
def marchenko_pastur_pdf(x, Q, sigma=1):
    y = 1 / Q
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)  # Largest eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)  # Smallest eigenvalue
    return (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a)) * (0 if (x > b or x < a) else 1)


def compare_eigenvalue_distribution(correlation_matrix, Q, sigma=1, set_autoscale=True, show_top=True):
    e, _ = np.linalg.eig(correlation_matrix)  # Correlation matrix is Hermitian, so this is faster
    # than other variants of eig

    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = 50
    if not show_top:
        # Clear top eigenvalue from plot
        e = e[e <= x_max + 1]
    ax.hist(e, density=True, bins=50)  # Histogram the eigenvalues
    ax.set_autoscale_on(set_autoscale)

    # Plot the theoretical density
    f = np.vectorize(lambda x: marchenko_pastur_pdf(x, Q, sigma=sigma))

    x_min = .0001 if np.power(sigma * (1 - np.sqrt(1 / Q)), 2) < .0001 else np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    x_max = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)

    x = np.linspace(x_min, x_max, 5000)
    ax.plot(x, f(x), linewidth=4, color='r')
    plt.show()


# Create the correlation matrix and find the eigenvalues
N = 500
T = 1000
X = np.random.normal(0, 1, size=(N, T))
cor = np.corrcoef(X)
Q = T / N
compare_eigenvalue_distribution(cor, Q)