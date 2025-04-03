import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

FEATURES_DIR = "data/features"  # Use relative path from the current working directory

def load_features():
    X, y = [], []

    for file in sorted(os.listdir(FEATURES_DIR)):  # Ensure sorted order
        if file.endswith("_features.npy"):
            file_path = os.path.join(FEATURES_DIR, file)
            features = np.load(file_path)

            # Extract subject ID from the filename (e.g., S001_R01_features.npy)
            subject_id = int(file.split("_")[0].replace("S", ""))  # Extracts '001' -> 1
            y.append(subject_id)
            X.append(features)

    y = np.array(y)
    print("Class distribution:", Counter(y))  # Check class distribution
    return np.array(X), y

# Load dataset
X, y = load_features()

# Handle missing values
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# Flatten feature vectors if necessary
if len(X.shape) > 2:
    X = X.reshape(X.shape[0], -1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Print class distribution
print("Class distribution in training set:", Counter(y_train))
print("Class distribution in test set:", Counter(y_test))

# Define SVM with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Train best model
best_svm = grid_search.best_estimator_
best_svm.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(best_svm, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Predict and evaluate
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
