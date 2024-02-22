import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preprocessing
mnist_data = pd.read_csv('')
features = mnist_data.iloc[:, :-1].values
labels = mnist_data.iloc[:, -1].values

# Mean Subtraction
mu = np.mean(features, axis=0)
X_centered = features - mu

# Covariance Matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]
cumulative_variance = np.cumsum(eigenvalues_sorted)
total_variance = cumulative_variance[-1]
variance_ratio = cumulative_variance / total_variance

# Eigenvalues that capture 99% of the variance
num_components_99 = np.argmax(variance_ratio >= 0.99) + 1

# Reconstruction Error
def reconstruct_data(n_components, eigenvectors, X_centered):
    projection = np.dot(X_centered, eigenvectors[:, :n_components])
    reconstruction = np.dot(projection, eigenvectors[:, :n_components].T)
    return reconstruction
reconstruction_errors = {}
for digit in np.unique(labels):
    digit_mask = labels == digit
    X_digit = X_centered[digit_mask]
    reconstruction_errors[digit] = []
    for i in range(1, num_components_99 + 1):
        X_reconstructed = reconstruct_data(i, eigenvectors_sorted, X_digit)
        error = np.mean(np.square(X_digit - X_reconstructed))
        reconstruction_errors[digit].append(error)
plt.figure(figsize=(10, 8))
for digit in reconstruction_errors:
    plt.plot(range(1, num_components_99 + 1), reconstruction_errors[digit], label=f'Digit {int(digit)}')

plt.title('Average Reconstruction Error as a Function of Eigenvectors')
plt.xlabel('Number of Eigenvectors')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.grid(True)
plt.savefig('MNIST_Eigen.png', dpi=300)
plt.show()
print(f"Number of eigenvectors needed to capture 99% of the variance: {num_components_99}")
