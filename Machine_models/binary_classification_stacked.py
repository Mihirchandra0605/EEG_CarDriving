import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Load full dataset
# --------------------
X = np.load('../Feature_Extraction_Result_files/npy_files_lda/X_full_lda.npy')
y = np.load('../Feature_Extraction_Result_files/npy_files_lda/y_full_lda.npy')


# --------------------
# Mask for binary classification: Right Fist (1) and Both Feet (3)
# --------------------
mask = np.logical_or(y == 1, y == 3)
X_binary = X[mask]
y_binary = y[mask]

y_binary = np.where(y_binary == 1, 0, 1)

# --------------------
# Train-test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# --------------------
# Feature Scaling
# --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------
# Classifier setups
# --------------------

# Random Forest + GridSearchCV
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1, verbose=1)

# SVM + GridSearchCV
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}
svm = SVC(probability=True)
svm_cv = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1, verbose=1)

# XGBoost (tuned)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.1, max_depth=6)

# --------------------
# Train models
# --------------------
print("Training Random Forest...")
rf_cv.fit(X_train, y_train)

print("Training SVM...")
svm_cv.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)

# --------------------
# Ensemble (Voting Classifier)
# --------------------
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_cv.best_estimator_),
        ('svm', svm_cv.best_estimator_),
        ('xgb', xgb)
    ],
    voting='soft'
)
print("Training Voting Classifier...")
voting_clf.fit(X_train, y_train)

# --------------------
# Evaluate all models
# --------------------
models = {
    'Random Forest': rf_cv.best_estimator_,
    'SVM': svm_cv.best_estimator_,
    'XGBoost': xgb,
    'Ensemble': voting_clf
}

for name, model in models.items():
    print(f"\n==== {name} ====")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
