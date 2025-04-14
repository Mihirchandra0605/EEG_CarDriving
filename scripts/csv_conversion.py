import os
import numpy as np
import pandas as pd

FEATURES_DIR = "../data/features"
CSV_DIR = "..data/csv_features"  # Directory to save CSV files
os.makedirs(CSV_DIR, exist_ok=True)  # Create directory if it doesn't exist

for file in sorted(os.listdir(FEATURES_DIR)):
    if file.endswith("_features.npy"):
        file_path = os.path.join(FEATURES_DIR, file)
        
        # Load the numpy file
        data = np.load(file_path)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_filename = os.path.join(CSV_DIR, file.replace(".npy", ".csv"))
        df.to_csv(csv_filename, index=False)
        
        print(f"Saved {csv_filename}")
