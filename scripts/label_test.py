import pickle
import os

sample_path = '../data/features/S001R03_features.pkl' 

with open(sample_path, 'rb') as f:
    data = pickle.load(f)
    print("Keys in the sample .pkl file:", data.keys())
