import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.load("../data/sorted_data/S15/S015R10_preprocessed.npy")
pca = PCA(n_components=data.shape[1])
pca.fit(data)

print(f"Variance explained by first {data.shape[1]} components: {sum(pca.explained_variance_ratio_):.2f}")




sample_data = np.load("../data/sorted_data/S1/S001R02_preprocessed.npy")[:1000]  # First 1000 samples
plt.plot(sample_data)
plt.title("Preprocessed EEG Sample")
plt.show()