import numpy as np
import mne
import pywt
import pickle
from scipy.stats import skew, kurtosis, entropy
import os
from pathlib import Path
from mne.annotations import Annotations


FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# === Function to extract bandpower ===
def extract_bandpower(epochs, sfreq):
    psds_obj = epochs.compute_psd(method='welch', fmin=8, fmax=30, n_fft=256)
    psds = psds_obj.get_data()
    psds_mean = np.mean(psds, axis=-1)  # Average across frequency bins
    bandpower = np.mean(psds_mean, axis=0)  # Average across epochs
    return bandpower

def extract_time_features(epoch_data):
    feats = []
    for epoch in epoch_data:
        mean = epoch.mean(axis=1)
        std = epoch.std(axis=1)
        sk = skew(epoch, axis=1)
        kurt_vals = kurtosis(epoch, axis=1)
        feats.append(np.concatenate([mean, std, sk, kurt_vals], axis=0))
    return np.array(feats)

def extract_hjorth(epoch_data):
    feats = []
    for epoch in epoch_data:
        activity = np.var(epoch, axis=1)
        mobility = np.sqrt(np.var(np.diff(epoch, axis=1), axis=1) / activity)
        complexity = np.sqrt(np.var(np.diff(np.diff(epoch, axis=1), axis=1), axis=1) /
                             np.var(np.diff(epoch, axis=1), axis=1)) / mobility
        feats.append(np.concatenate([activity, mobility, complexity], axis=0))
    return np.array(feats)

def extract_entropy(epoch_data):
    feats = []
    for epoch in epoch_data:
        ent = [entropy(np.abs(ch) / np.sum(np.abs(ch))) for ch in epoch]
        feats.append(np.array(ent))
    return np.array(feats)

def extract_wavelet(epoch_data, wavelet='db4'):
    feats = []
    for epoch in epoch_data:
        ch_feats = []
        for ch in epoch:
            coeffs = pywt.wavedec(ch, wavelet, level=5)
            energies = [np.sum(np.square(c)) for c in coeffs]
            ch_feats.append(energies)
        feats.append(np.array(ch_feats).flatten())
    return np.array(feats)

def load_annotations(edf_path):
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, stim_channel=None, verbose=False)
    annotations = raw.annotations
    events, event_id = mne.events_from_annotations(raw)
    return events, event_id

def main():
    preprocessed_root = '../data/Preprocessed_data1'
    edf_root = '../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files'

    subjects = sorted(os.listdir(preprocessed_root))
    for subject in subjects:
        subject_path = os.path.join(preprocessed_root, subject)
        if not os.path.isdir(subject_path):
            continue

        print(f"\n=== Processing Subject: {subject} ===")
        print(f"Preprocessed directory: {subject_path}")

        edf_subject_path = os.path.join(edf_root, subject)
        print(f"EDF directory:         {edf_subject_path}")

        for npy_file in sorted(os.listdir(subject_path)):
            if not npy_file.endswith('.npy'):
                continue

            print(f"\n🔹 Processing file: {npy_file}")
            base_name = npy_file.replace('.npy', '')
            edf_file = base_name + '.edf'
            edf_path = os.path.join(edf_subject_path, edf_file)
            print(f"  ➔ Expected EDF file: {edf_path}")
            print(f"  ➔ EDF Exists? {os.path.exists(edf_path)}")

            if not os.path.exists(edf_path):
                print(f" Matching EDF file not found for {npy_file}. Skipping.")
                continue

            eeg_data = np.load(os.path.join(subject_path, npy_file))
            print(f"✅ Loaded EEG data: shape = {eeg_data.shape}")

            # Check if transpose is needed
            if eeg_data.shape[0] > 1000:  # too many "channels"
                print("⚠️ EEG data likely needs transposing. Applying transpose...")
                eeg_data = eeg_data.T
                print(f"🔁 Transposed EEG data shape: {eeg_data.shape}")


            raw_edf = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)
            sfreq = raw_edf.info['sfreq']
            full_ch_names = raw_edf.info['ch_names']

            # Debugging: match shape
            n_channels, n_samples = eeg_data.shape
            print(f"EDF has {len(full_ch_names)} channels; EEG data has {n_channels} channels.")

            if n_channels != len(full_ch_names):
                print(" Channel count mismatch. Trying to slice channel names accordingly...")
                if n_channels < len(full_ch_names):
                    ch_names = [f'PC{i+1}' for i in range(n_channels)]

                else:
                    print(" EEG data has more channels than the EDF file supports. Skipping.")
                    continue
            else:
                ch_names = full_ch_names

            try:
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
                raw_array = mne.io.RawArray(eeg_data, info)
            except Exception as e:
                print(f" Failed to create RawArray: {e}")
                continue

            original_annotations = raw_edf.annotations
            annotations = Annotations(
                onset=original_annotations.onset,
                duration=original_annotations.duration,
                description=original_annotations.description,
                orig_time=None  
            )
            raw_array.set_annotations(annotations)

            print("Used Annotations descriptions:", np.unique(annotations.description))

            try:
                events, _ = mne.events_from_annotations(raw_array)
                event_id = {'T0': 1, 'T1': 2, 'T2': 3}
                epochs = mne.Epochs(raw_array, events, event_id=event_id, tmin=-0.5, tmax=4.0,
                                    baseline=None, preload=True, reject_by_annotation=True)
                print(" Drop log for", npy_file, ":", epochs.drop_log)

                if len(epochs) == 0:
                    print(f"⚠️ All epochs were dropped for {npy_file}. Skipping.")
                    continue

                # === Extract all features ===
                bandpower = extract_bandpower(epochs, sfreq)
                time_feats = extract_time_features(epochs.get_data())
                hjorth_feats = extract_hjorth(epochs.get_data())
                entropy_feats = extract_entropy(epochs.get_data())
                wavelet_feats = extract_wavelet(epochs.get_data())

                print(f"✅ Bandpower shape: {bandpower.shape}, Sample values: {bandpower[:5]}")

                # === Combine all features ===
                features = {
                    'bandpower': bandpower,
                    'time': time_feats,
                    'hjorth': hjorth_feats,
                    'entropy': entropy_feats,
                    'wavelet': wavelet_feats
                }

                # === Save features as .pkl ===
                os.makedirs('../data/features', exist_ok=True)
                feature_save_path = os.path.join('../data/features', base_name + '_features.pkl')
                with open(feature_save_path, 'wb') as f:
                    pickle.dump(features, f)

                print(f" Saved features to {feature_save_path}")

            except Exception as e:
                print(f" Error during epoching or feature extraction: {e}")
                continue


if __name__ == '__main__':
    main()
