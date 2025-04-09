import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap  # Make sure you have `umap-learn` installed

# Load original data
X_train = np.load('../src/X_train.npy')
y_train = np.load('../src/y_train.npy')
X_test = np.load('../src/X_test.npy')
y_test = np.load('../src/y_test.npy')

# Remove class 4 from both train and test
train_mask = y_train != 4
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]

test_mask = y_test != 4
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

# Train the model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_filtered, y_train_filtered)

y_pred = model.predict(X_test_filtered)

# Classification report
print("Classification Report:")
print(classification_report(y_test_filtered, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test_filtered, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=np.unique(y_test_filtered), yticklabels=np.unique(y_test_filtered))
plt.title("Confusion Matrix (4-Class)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Feature scaling (important for t-SNE & UMAP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_filtered)
y_labels = y_train_filtered

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for label in np.unique(y_labels):
    idx = y_labels == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f'Class {label}', alpha=0.6)
plt.legend()
plt.title('t-SNE Projection of EEG Features')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- UMAP ---
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for label in np.unique(y_labels):
    idx = y_labels == label
    plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=f'Class {label}', alpha=0.6)
plt.legend()
plt.title('UMAP Projection of EEG Features')
plt.grid(True)
plt.tight_layout()
plt.show()
