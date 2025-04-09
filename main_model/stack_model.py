from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load LDA reduced features
X = np.load('../src/X_full_lda.npy')
y = np.load('../src/y_full_lda.npy')


# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale (for SVM & Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base learners
base_learners = [
    ('lda', LinearDiscriminantAnalysis()),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf'))
]

# Define meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# Define stacking classifier
stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    passthrough=True, # also include original features
    cv=5
)

# Define parameter grid
param_grid = {
    'final_estimator__C': [0.1, 1, 10],
    'final_estimator__solver': ['lbfgs'],
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [None, 10],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto']
}

# GridSearchCV
grid = GridSearchCV(estimator=stack, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Best Params:", grid.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
