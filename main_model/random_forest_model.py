import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import os
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score  # Uncomment if using CV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def extract_label_from_filename(filename):
    run_str = filename.split('R')[-1].split('_')[0]  # Updated split logic
    run_num = int(run_str)

    if run_num in [3, 4, 7, 8, 11, 12]:
        return {
            3: 0,  # Left fist
            4: 1,  # Right fist
            7: 0,
            8: 1,
            11: 0,
            12: 1
        }[run_num]
    
    elif run_num in [5, 6, 9, 10, 13, 14]:
        return {
            5: 2,  # Both fists
            6: 3,  # Both feet
            9: 2,
            10: 3,
            13: 2,
            14: 3
        }[run_num]

    else:
        return -1


# Step 1: Load and process features
def load_features():
    X_all = []
    y_all = []

    features_dir = '../data/features'  # Update if your path is different
    expected_length = None  # Track correct feature vector length

    for filename in os.listdir(features_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(features_dir, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

                try:
                    combined_features = np.concatenate([
                        data['bandpower'].flatten(),
                        data['time'].flatten(),
                        data['hjorth'].flatten(),
                        data['entropy'].flatten(),
                        data['wavelet'].flatten()
                    ])

                    # Check if shape is consistent
                    if expected_length is None:
                        expected_length = combined_features.shape[0]
                    elif combined_features.shape[0] != expected_length:
                        print(f" Skipping {filename} due to inconsistent feature length: {combined_features.shape[0]} != {expected_length}")
                        continue

                    label = extract_label_from_filename(filename)
                    if label == -1:
                        print(f" Skipping {filename} due to unknown label.")
                        continue

                    X_all.append(combined_features)
                    y_all.append(label)

                except Exception as e:
                    print(f" Error loading {filename}: {e}")

    return np.array(X_all), np.array(y_all)


# Step 2: Load data
X, y = load_features()
print(f" Total Samples: {len(y)}")
print(" Class Distribution:", Counter(y))


# # Split your features and labels
# X = data.drop('label', axis=1)
# y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize Random Forest with class_weight
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Handles class imbalance
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Print classification report
print(" Classification Report (Random Forest):\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="inferno")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()
