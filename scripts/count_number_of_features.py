from collections import Counter
import os, pickle

folder = "../data/features"
lengths = []

for f in os.listdir(folder):
    if f.endswith("_features.pkl"):
        with open(os.path.join(folder, f), "rb") as file:
            data = pickle.load(file)
            lengths.append(len(data))

print(Counter(lengths))
