import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# Load the data
data_path = 'C:/Users/cahendren/Data/Sims/MNIST_digits0-1-2.csv'  # Update this path as necessary
column_names = [f'pixel{i}' for i in range(784)] + ['label']
data = pd.read_csv(data_path, header=None, names=column_names)

# Splitting the dataset into training and test sets for each class
train_sets = []
test_sets = []
for digit in [0, 1, 2]:
    digit_data = data[data['label'] == digit]
    train, test = train_test_split(digit_data, test_size=0.2, random_state=42)
    train_sets.append(train)
    test_sets.append(test)

train_data = pd.concat(train_sets)
test_data = pd.concat(test_sets)

X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Apply LDA to project the data into two dimensions
lda = LDA(n_components=2)
X_lda_train = lda.fit_transform(X_train, y_train)
X_lda_test = lda.transform(X_test)

# Visualization
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green']
targets = [0, 1, 2]
labels = ['Train 0', 'Train 1', 'Train 2', 'Test 0', 'Test 1', 'Test 2']
markers = ['o', '^']  # Different markers for train and test

# Plot training data
for color, target in zip(colors, targets):
    plt.scatter(X_lda_train[y_train == target, 0], X_lda_train[y_train == target, 1], color=color, marker=markers[0], alpha=0.5, label=f'Train {target}')

# Plot test data
for color, target in zip(colors, targets):
    plt.scatter(X_lda_test[y_test == target, 0], X_lda_test[y_test == target, 1], color=color, marker=markers[1], alpha=0.5, label=f'Test {target}')

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA Projection of MNIST Digits 0, 1, 2')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.show()
