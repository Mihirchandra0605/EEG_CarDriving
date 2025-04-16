from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import pandas as pd


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Load LDA reduced features
X = np.load('../Feature_Extraction_Result_files/npy_files_lda/X_full_lda.npy')
y = np.load('../Feature_Extraction_Result_files/npy_files_lda/y_full_lda.npy')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Set up param grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Evaluate
y_pred = best_rf.predict(X_test)
print(" Best Accuracy:", best_rf.score(X_test, y_test))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: print best parameters
print("\n🔧 Best Parameters:", grid_search.best_params_)

# Plot confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()



# # Load LDA-transformed features and true labels
# X = np.load('../src/X_full_lda.npy')     # shape: (n_samples, 3)
# y_true = np.load('../src/y_full_lda.npy')  # shape: (n_samples,)

# # Predict using best RF
# y_pred = best_rf.predict(X)

# # Color by correctness
# correct = y_pred == y_true
# colors = np.where(correct, 'green', 'red')

# # 3D scatter plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, alpha=0.6, s=12)

# ax.set_title("Correct (Green) vs Incorrect (Red) Predictions in LDA Space")
# ax.set_xlabel("LDA Component 1")
# ax.set_ylabel("LDA Component 2")
# ax.set_zlabel("LDA Component 3")

# plt.show()
