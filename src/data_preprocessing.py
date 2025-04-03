import os
import numpy as np
import mne
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR = "../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files" 
PREPROCESSED_DATA_DIR = "../data/Preprocessed_data"

# Ensure preprocessed data directory exists
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

# Define bandpass filter range
LOW_CUTOFF = 4  # 4Hz
HIGH_CUTOFF = 30  # 30Hz

def preprocess_eeg(file_path):
    """Load, filter, apply ICA & PCA, normalize, and save EEG data."""
    
    print(f"Processing: {file_path} ...")
    
    # Load EEG data
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_types(eeg=True)  # Keep only EEG channels
    raw.resample(160)  # Ensure uniform sampling rate (if needed)

    # 1) Bandpass Filter (4-30Hz)
    raw.filter(LOW_CUTOFF, HIGH_CUTOFF, fir_design='firwin')

    # 2️) Notch Filter (Remove 50Hz powerline noise)
    raw.notch_filter(freqs=50)

    # 3️) Apply ICA (Artifact Removal)
    ica = ICA(n_components=25, random_state=42)
    ica.fit(raw)
    raw = ica.apply(raw)

    # 4️) Convert EEG to Numpy Array
    eeg_data = raw.get_data()

    # 5️) Apply PCA (Dimensionality Reduction)
    pca = PCA(n_components=20)  # Keep 20 components
    eeg_pca = pca.fit_transform(eeg_data.T)  # Transpose because PCA expects (samples, features)

    # 6️) Normalize EEG Data
    scaler = StandardScaler()
    eeg_pca_normalized = scaler.fit_transform(eeg_pca)

    # 7️) Save Preprocessed Data
    save_path = os.path.join(PREPROCESSED_DATA_DIR, os.path.basename(file_path).replace('.edf', '_preprocessed.npy'))
    np.save(save_path, eeg_pca_normalized)
    
    print(f"Preprocessed EEG saved to {save_path}")

# Process all files in raw_data directory
for subject_folder in os.listdir(RAW_DATA_DIR):
    subject_path = os.path.join(RAW_DATA_DIR, subject_folder)
    
    if os.path.isdir(subject_path):
        for file in os.listdir(subject_path):
            if file.endswith(".edf"):
                file_path = os.path.join(subject_path, file)
                preprocess_eeg(file_path)

print("All EEG data preprocessed and saved!")