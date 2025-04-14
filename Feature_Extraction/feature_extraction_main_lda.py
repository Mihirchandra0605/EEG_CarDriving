import os
import numpy as np
import pywt
import scipy.stats as stats
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Only relevant EEG bands (based on preprocessing: bandpass 8–30 Hz)
EEG_BANDS = {
    'alpha': (8, 13),
    'beta': (13, 30)
}

# Hjorth parameter helpers
def compute_hjorth(eeg):
    first_deriv = np.diff(eeg)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(eeg)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0

    return activity, mobility, complexity

# Bandpower using Welch’s method
def compute_bandpower(data, sf, band):
    fmin, fmax = band
    freqs, psd = welch(data, sf)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])

# Wavelet energy
def compute_wavelet_energy(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energies = [np.sum(np.square(c)) for c in coeffs]
    return energies

def extract_features_from_segment(segment, sfreq=160):
    """
    Extracts features from a segment of shape (21, time_samples).
    Each channel is processed individually for time-domain, Hjorth, bandpower (alpha/beta), and wavelet energy.
    """
    features = []
    for ch_data in segment:
        ch_feats = []

        # Time-domain features
        ch_feats.extend([
            np.mean(ch_data),
            np.std(ch_data),
            np.min(ch_data),
            np.max(ch_data),
            stats.skew(ch_data),
            stats.kurtosis(ch_data)
        ])

        # Hjorth parameters
        ch_feats.extend(compute_hjorth(ch_data))

        # Alpha and beta bandpower
        for band in EEG_BANDS.values():
            ch_feats.append(compute_bandpower(ch_data, sfreq, band))

        # Wavelet energy
        ch_feats.extend(compute_wavelet_energy(ch_data))

        features.extend(ch_feats)

    return features

def load_data_and_extract_features(data_dir, apply_pca=True, n_components=100, remove_rest=True):
    X = []
    y = []

    label_map = {
        'left_fist': 0,
        'right_fist': 1,
        'both_fists': 2,
        'both_feet': 3,
        'rest': 4
    }

    print(" Extracting features from preprocessed EEG segments...")
    for subject in tqdm(os.listdir(data_dir)):
        subj_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subj_path):
            continue

        for file in os.listdir(subj_path):
            if not file.endswith('.npy'):
                continue

            # Extract label string
            label_parts = file.replace('.npy', '').split('_')
            label_str = '_'.join(label_parts[4:])

            if label_str not in label_map:
                print(f" Skipping unknown label in file: {file}")
                continue

            # Skip rest class if remove_rest is True
            if remove_rest and label_str == 'rest':
                continue

            segment = np.load(os.path.join(subj_path, file))
            features = extract_features_from_segment(segment)
            X.append(features)
            y.append(label_map[label_str])

    X = np.array(X)
    y = np.array(y)

    print(f" Feature matrix shape: {X.shape}, Label distribution: {np.unique(y, return_counts=True)}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for dimensionality reduction
    if apply_pca:  # we’ll keep this flag for flexibility
        print(f"Applying LDA for supervised dimensionality reduction (max 3 components)...")
        lda = LDA(n_components=3)
        X_reduced = lda.fit_transform(X_scaled, y)
        return X_reduced, y

    return X_scaled, y

if __name__ == '__main__':
    DATA_DIR = '../data/Preprocessed_data_main2'  

    # Use LDA by setting apply_pca=True 
    X, y = load_data_and_extract_features(DATA_DIR, apply_pca=True, remove_rest=True)

    np.save('X_full_lda.npy', X)
    np.save('y_full_lda.npy', y)
    print(" Saved features and labels.")
