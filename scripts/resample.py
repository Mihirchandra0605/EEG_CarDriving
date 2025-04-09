from sklearn.utils import resample
import numpy as np
import os

X_train = np.load('../src/X_train.npy')
y_train = np.load('../src/y_train.npy')

# Combine features and labels
Xy_train = list(zip(X_train, y_train))

# Separate by class
separated = {label: [] for label in set(y_train)}
for x, y in Xy_train:
    separated[y].append((x, y))

# Downsample all classes to the size of the smallest class
min_class_len = min(len(v) for v in separated.values())
balanced_data = []

for label, items in separated.items():
    balanced_items = resample(items, n_samples=min_class_len, random_state=42)
    balanced_data.extend(balanced_items)

# Unzip and convert to numpy arrays
X_balanced, y_balanced = zip(*balanced_data)
X_balanced = np.array(X_balanced)
y_balanced = np.array(y_balanced)

# ✅ Save the balanced arrays
np.save('../src/X_train_balanced.npy', X_balanced)
np.save('../src/y_train_balanced.npy', y_balanced)

print("✅ Balanced training data saved.")
print(f"Shape: X_balanced = {X_balanced.shape}, y_balanced = {y_balanced.shape}")
