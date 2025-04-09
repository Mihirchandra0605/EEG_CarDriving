import pickle
import pandas as pd
import numpy as np

with open('../data/features/S001R03_features.pkl', 'rb') as f:
    data = pickle.load(f)

# Get the minimum length among all keys
min_len = min([v.shape[0] if isinstance(v, np.ndarray) else len(v) for v in data.values()])

flat_data = {}

for key, value in data.items():
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            for i in range(value.shape[1]):
                flat_data[f'{key}_{i}'] = value[:min_len, i]
        else:
            flat_data[key] = value[:min_len]
    elif isinstance(value, list):
        flat_data[key] = value[:min_len]
    else:
        # Convert other types to list or skip
        pass

df = pd.DataFrame(flat_data)
df.to_csv('output.csv', index=False)
print(" 'output.csv'")


