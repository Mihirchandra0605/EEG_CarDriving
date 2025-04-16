import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# LDA-transformed features and labels
X = np.load('../Feature_Extraction_Result_files/npy_files_lda/X_full_lda.npy') 
y = np.load('../Feature_Extraction_Result_files/npy_files_lda/y_full_lda.npy')

# Map class labels to colors and names
label_names = {
    0: 'Left Fist',
    1: 'Right Fist',
    2: 'Both Fists',
    3: 'Both Feet'
}
colors = ['r', 'g', 'b', 'purple']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(y):
    idx = y == label
    ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], 
               c=colors[label], label=label_names[label], s=15, alpha=0.6)

ax.set_title('3D LDA Feature Space')
ax.set_xlabel('LDA Component 1')
ax.set_ylabel('LDA Component 2')
ax.set_zlabel('LDA Component 3')
ax.legend()
plt.tight_layout()
plt.show()
