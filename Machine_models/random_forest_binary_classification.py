from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np

X = np.load('../Feature_Extraction_Result_files/npy_files_lda/X_full_lda.npy')
y = np.load('../Feature_Extraction_Result_files/npy_files_lda/y_full_lda.npy')


# Suppose y contains labels: 0 = Left Fist, 1 = Right Fist, 2 = Both Fists, 3 = Both Feet
mask = np.isin(y, [1, 3])  # Right Fist vs Both Feet
X_binary = X[mask]
y_binary = y[mask]

# Convert to binary labels: 0 and 1
y_binary = (y_binary == 3).astype(int)  # 1 for Both Feet, 0 for Right Fist


X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.20, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X_binary)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[y_binary==0, 0], X_pca_3d[y_binary==0, 1], X_pca_3d[y_binary==0, 2], label='Right Fist', alpha=0.6)
ax.scatter(X_pca_3d[y_binary==1, 0], X_pca_3d[y_binary==1, 1], X_pca_3d[y_binary==1, 2], label='Both Feet', alpha=0.6)
ax.set_title('3D PCA - Right Fist vs Both Feet')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.legend()
plt.show()
