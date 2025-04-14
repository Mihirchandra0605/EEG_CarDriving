import numpy as np
import matplotlib.pyplot as plt

# Load the files
X = np.load("../src/X_full_nolda.npy")
y = np.load("../src/y_full_nolda.npy")

# Print shape
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# X = X.reshape((X.shape[0], 3, 1))  # shape: (18367, time_steps=3, features=1)

# print("Reshaped X:", X.shape)