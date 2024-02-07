import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Preprocessing
data_path = ''
column_names = [f'pixel{i}' for i in range(784)] + ['label']
data = pd.read_csv(data_path, header=None, names=column_names)
X = data.drop('label', axis=1)
y = data['label']

# 2D
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], ['0', '1', '2']):
    plt.scatter(X_pca_2d[y == i, 0], X_pca_2d[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA Visualization of MNIST Digits 0, 1, 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('/MNIST_PCA_2D.png', dpi=300)
plt.show()

# 3D
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for color, i, target_name in zip(colors, [0, 1, 2], ['0', '1', '2']):
    ax.scatter(X_pca_3d[y == i, 0], X_pca_3d[y == i, 1], X_pca_3d[y == i, 2], color=color, lw=lw, label=target_name)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title('3D PCA Visualization of MNIST Digits 0, 1, 2')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.savefig('MNIST_PCA_3D.png', dpi=300)
plt.show()