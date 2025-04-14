from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load your LDA-reduced features
X_train = np.load('../src/X_train_lda.npy')
y_train = np.load('../src/y_train_lda.npy')
X_test = np.load('../src/X_test_lda.npy')
y_test = np.load('../src/y_test_lda.npy')

clf = SVC(kernel='rbf', C=1, gamma='scale')  # try tweaking C and gamma later
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
