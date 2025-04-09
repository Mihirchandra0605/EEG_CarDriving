import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Load your LDA-reduced features
X_train = np.load('../src/X_train_lda.npy')
y_train = np.load('../src/y_train_lda.npy')
X_test = np.load('../src/X_test_lda.npy')
y_test = np.load('../src/y_test_lda.npy')

# Initialize Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Cross-validation score on training data
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print(f"\n CV Accuracy (5-Fold): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
