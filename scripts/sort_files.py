import os
import shutil

# Define paths
preprocessed_dir = "../data/Preprocessed_data"
output_dir = "../data/sorted_data"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all .npy files
files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.npy')]

for file in files:
    # Extract subject number
    subject_id = file.split('R')[0][1:]  # Extracts number after 'S'
    subject_folder = os.path.join(output_dir, f"S{int(subject_id)}")  # S1, S2, ..., S109
    
    # Create subject folder if it doesn't exist
    os.makedirs(subject_folder, exist_ok=True)

    # Move file to its respective folder
    shutil.move(os.path.join(preprocessed_dir, file), os.path.join(subject_folder, file))

print("Files sorted successfully!")
