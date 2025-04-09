import numpy as np
y_train = np.load('../src/y_train_lda.npy')
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
