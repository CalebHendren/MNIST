import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load the COIL20 dataset from the uploaded file
mat = scipy.io.loadmat(r'C:\Users\caleb\Desktop\Assignment2\Assignment2\COIL20.mat')
X = mat['X']  # Correct key for features
y = mat['Y']  # Correct key for labels

# Ensure y is a 1-dimensional array
y = y.ravel()

# Step 2: Partition Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Fit PCA and LDA, Project Data
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize PCA and LDA
pca = PCA(n_components=19)  # Adjust based on the dataset
lda = LDA(n_components=min(len(np.unique(y)) - 1, X.shape[1]))  # Min of classes-1 and features

# Fit and transform the data
X_train_pca = pca.fit_transform(X_train_scaled)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)

X_test_pca = pca.transform(X_test_scaled)
X_test_lda = lda.transform(X_test_scaled)

# Step 4: Calculate 5NN Accuracy and Plot
pca_accuracies = []
lda_accuracies = []

for dim in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=5)

    # For PCA
    knn.fit(X_train_pca[:, :dim], y_train)
    pca_accuracies.append(accuracy_score(y_test, knn.predict(X_test_pca[:, :dim])))

    # For LDA (considering dimensionality limit)
    if dim <= lda.n_components:
        knn.fit(X_train_lda[:, :dim], y_train)
        lda_accuracies.append(accuracy_score(y_test, knn.predict(X_test_lda[:, :dim])))
    else:
        lda_accuracies.append(lda_accuracies[-1])  # Repeat the last accuracy as max dimension reached

# Plot accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, 20), pca_accuracies, label='PCA')
plt.plot(range(1, 20), lda_accuracies, label='LDA')
plt.xlabel('Number of Dimensions')
plt.ylabel('5NN Accuracy')
plt.title('5NN Accuracy vs. Number of Dimensions')
plt.legend()
plt.show()

# Ensure there are at least 3 components for PCA and LDA
n_components = min(3, pca.n_components_, lda.n_components)

# Step 5: Combined Scatter Plot in Three-Dimensional Space for PCA and LDA
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot PCA
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap='viridis', marker='o', label='PCA')

# Plot LDA - Ensure LDA has at least 3 components, otherwise adjust dimensions
if lda.n_components >= 3:
    ax.scatter(X_train_lda[:, 0], X_train_lda[:, 1], X_train_lda[:, 2], c=y_train, cmap='plasma', marker='^', label='LDA')
else:
    # Adjust for LDA with fewer than 3 dimensions
    print(f"LDA has fewer than 3 components ({lda.n_components}), adjusting plot dimensions.")
    if lda.n_components == 2:
        ax.scatter(X_train_lda[:, 0], X_train_lda[:, 1], np.zeros(X_train_lda.shape[0]), c=y_train, cmap='plasma', marker='^', label='LDA')
    elif lda.n_components == 1:
        ax.scatter(X_train_lda[:, 0], np.zeros(X_train_lda.shape[0]), np.zeros(X_train_lda.shape[0]), c=y_train, cmap='plasma', marker='^', label='LDA')

ax.set_title('PCA vs LDA 3D Scatter Plot')
ax.set_xlabel('First Dimension')
ax.set_ylabel('Second Dimension')
ax.set_zlabel('Third Dimension')
ax.legend()

plt.show()