import numpy as np
from collections import Counter

y = np.load('y_labels.npy')
print("Label distribution:", Counter(y))
