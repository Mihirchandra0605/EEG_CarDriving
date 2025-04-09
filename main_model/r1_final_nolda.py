import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load LDA-processed features
X = np.load('../src/X_full_nolda.npy')
y = np.load('../src/y_full_nolda.npy')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Choose classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf = SVC(kernel='rbf', C=1, gamma='scale')  # Uncomment to try SVM

# Train the model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f" Accuracy: {acc}\n")

print(" Classification Report:")
print(classification_report(y_test, y_pred))

print(" Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f" CV Accuracy (5-Fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
