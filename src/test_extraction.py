import os
import numpy as np
import pywt
import scipy.stats as stats
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# EEG bands (in Hz)
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
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

def extract_features_from_segment(segment, sfreq=160):  # update sfreq if different
    features = []
    for ch_data in segment:
        ch_feats = []

        # Time-domain
        ch_feats.extend([
            np.mean(ch_data),
            np.std(ch_data),
            np.min(ch_data),
            np.max(ch_data),
            stats.skew(ch_data),
            stats.kurtosis(ch_data)
        ])

        # Hjorth
        hjorth = compute_hjorth(ch_data)
        ch_feats.extend(hjorth)

        # Bandpower
        for band in EEG_BANDS.values():
            bp = compute_bandpower(ch_data, sfreq, band)
            ch_feats.append(bp)

        # Wavelet energy
        wave_energy = compute_wavelet_energy(ch_data)
        ch_feats.extend(wave_energy)

        features.extend(ch_feats)

    return features

def test_specific_files(file_paths, label_map):
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    for path in file_paths:
        file_name = os.path.basename(path)
        label_parts = file_name.replace('.npy', '').split('_')
        label_str = '_'.join(label_parts[4:])



        if label_str not in label_map:
            print(f" Unknown label format in {file_name}")
            continue

        expected_label = label_map[label_str]
        data = np.load(path)
        features = extract_features_from_segment(data)

        print(f"📄 File: {file_name}")
        print(f"   ➤ Expected Label: {expected_label} ({label_str})")
        print(f"   ➤ Feature length: {len(features)}\n")

if __name__ == '__main__':
    test_files = [
        '../data/Preprocessed_data_main/S001/run_03_seg_000_rest.npy',
        '../data/Preprocessed_data_main/S001/run_03_seg_001_right_fist.npy',
        '../data/Preprocessed_data_main/S001/run_03_seg_005_left_fist.npy'
    ]

    label_map = {
        'left_fist': 0, 
        'right_fist': 1,
        'both_fists': 2,
        'both_feet': 3,
        'rest': 4
    }

    test_specific_files(test_files, label_map)
