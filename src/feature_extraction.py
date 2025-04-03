import os
import numpy as np
import pywt
import scipy.signal as signal
from scipy.fftpack import fft

# Define paths
data_dir = "../data/sorted_data"
features_dir = "../data/features"
os.makedirs(features_dir, exist_ok=True)  # Ensure features folder exists

# EEG parameters
fs = 160  # Sampling frequency
channels = 64  # 64 EEG channels

# Loop through subjects S001 to S109
for subject in range(1, 110):
    subject_folder = f"S{subject:03d}"
    subject_path = os.path.join(data_dir, subject_folder)
    
    if not os.path.exists(subject_path):
        print(f"Skipping {subject_folder}, folder not found.")
        continue
    
    print(f"Processing {subject_folder}...")
    
    for run in range(1, 15):  # R01 to R14
        file_name = f"S{subject:03d}R{run:02d}_preprocessed.npy"
        file_path = os.path.join(subject_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"Skipping {file_name}, file not found.")
            continue
        
        # Load EEG data
        data = np.load(file_path)  # Shape: (channels, samples)
        if data.size == 0:
            print(f"Skipping {file_name}, empty file.")
            continue

        # Feature Extraction
        features = []
        for ch in range(channels):
            signal_data = data[ch, :]
            
            # FFT
            fft_values = np.abs(fft(signal_data))[:len(signal_data)//2]  # One-sided spectrum
            
            # Wavelet Transform
            max_level = pywt.dwt_max_level(len(signal_data), 'db4')  # Compute max decomposition level
            coeffs = pywt.wavedec(signal_data, 'db4', level=max_level)
            wavelet_features = np.hstack([np.mean(np.abs(c)) for c in coeffs])
            
            # Combine features
            channel_features = np.hstack([fft_values, wavelet_features])
            features.append(channel_features)
        
        features = np.array(features)  # Shape: (channels, feature_vector_size)
        
        # Save extracted features
        save_path = os.path.join(features_dir, f"{subject_folder}_R{run:02d}_features.npy")
        np.save(save_path, features)
        print(f"Saved features: {save_path}")

print("Feature extraction complete!")