# pca_preprocess.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your full dataset
X = np.load("../src/X_full_nolda.npy")
y = np.load("../src/y_full_nolda.npy")

# Normalize before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=100)  # or 50
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)

# # Save
# np.save("../src/X_pca.npy", X_pca)
# np.save("../src/y_pca.npy", y)

explained = np.cumsum(pca.explained_variance_ratio_)
print("Variance retained:", explained[-1])
