import numpy as np
import mne
import pywt
import pickle
import os
from scipy.stats import skew, kurtosis, entropy
from mne.annotations import Annotations

FREQ_BANDS = {
    'alpha': (8, 13),
    'beta': (13, 30)
}

def extract_bandpower(epochs):
    psds_obj = epochs.compute_psd(method='welch', fmin=8, fmax=30, n_fft=256)
    psds = psds_obj.get_data()
    psds_mean = np.mean(psds, axis=-1)
    bandpower = np.mean(psds_mean, axis=0)
    return bandpower

def extract_time_features(epoch_data):
    return np.array([
        np.concatenate([ep.mean(axis=1), ep.std(axis=1), skew(ep, axis=1), kurtosis(ep, axis=1)], axis=0)
        for ep in epoch_data
    ])

def extract_hjorth(epoch_data):
    feats = []
    for ep in epoch_data:
        activity = np.var(ep, axis=1)
        mobility = np.sqrt(np.var(np.diff(ep, axis=1), axis=1) / activity)
        complexity = np.sqrt(np.var(np.diff(np.diff(ep, axis=1), axis=1), axis=1) /
                             np.var(np.diff(ep, axis=1), axis=1)) / mobility
        feats.append(np.concatenate([activity, mobility, complexity], axis=0))
    return np.array(feats)

def extract_entropy(epoch_data):
    return np.array([
        np.array([entropy(np.abs(ch) / np.sum(np.abs(ch))) for ch in ep])
        for ep in epoch_data
    ])

def extract_wavelet(epoch_data, wavelet='db4'):
    feats = []
    for ep in epoch_data:
        ch_feats = []
        for ch in ep:
            coeffs = pywt.wavedec(ch, wavelet, level=5)
            energies = [np.sum(np.square(c)) for c in coeffs]
            ch_feats.append(energies)
        feats.append(np.array(ch_feats).flatten())
    return np.array(feats)

def process_subject(subject_path, edf_subject_path):
    for npy_file in sorted(os.listdir(subject_path)):
        if not npy_file.endswith('.npy'):
            continue

        base_name = npy_file.replace('.npy', '')
        edf_file = base_name + '.edf'
        edf_path = os.path.join(edf_subject_path, edf_file)

        if not os.path.exists(edf_path):
            print(f" Missing EDF for {npy_file}. Skipping.")
            continue

        eeg_data = np.load(os.path.join(subject_path, npy_file))
        if eeg_data.shape[0] > 1000:
            eeg_data = eeg_data.T

        raw_edf = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)
        sfreq = raw_edf.info['sfreq']
        full_ch_names = raw_edf.info['ch_names']

        n_channels, _ = eeg_data.shape
        if n_channels <= len(full_ch_names):
            ch_names = full_ch_names[:n_channels]
        else:
            ch_names = [f'PC{i+1}' for i in range(n_channels)]

        try:
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            raw_array = mne.io.RawArray(eeg_data, info)
        except Exception as e:
            print(f" RawArray creation failed: {e}")
            continue

        # Use original annotations
        original_annotations = raw_edf.annotations
        annotations = Annotations(
            onset=original_annotations.onset,
            duration=original_annotations.duration,
            description=original_annotations.description,
            orig_time=None
        )
        raw_array.set_annotations(annotations)

        try:
            events, _ = mne.events_from_annotations(raw_array)
            event_id = {'T0': 1, 'T1': 2, 'T2': 3}
            epochs = mne.Epochs(raw_array, events, event_id=event_id, tmin=-0.5, tmax=4.0,
                                baseline=None, preload=True, reject_by_annotation=True)
            if len(epochs) == 0:
                print(f" All epochs dropped for {npy_file}. Skipping.")
                continue

            bandpower = extract_bandpower(epochs)
            data = epochs.get_data()
            time_feats = extract_time_features(data)
            hjorth_feats = extract_hjorth(data)
            entropy_feats = extract_entropy(data)
            wavelet_feats = extract_wavelet(data)

            features = {
                'bandpower': bandpower,
                'time': time_feats,
                'hjorth': hjorth_feats,
                'entropy': entropy_feats,
                'wavelet': wavelet_feats
            }

            os.makedirs('../data/features', exist_ok=True)
            save_path = os.path.join('../data/features2', base_name + '_features.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(features, f)

            print(f" Features saved: {save_path}")

        except Exception as e:
            print(f" Processing failed for {npy_file}: {e}")

def main():
    preprocessed_root = '../data/Preprocessed_data1'
    edf_root = '../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files'

    subjects = sorted(os.listdir(preprocessed_root))
    for subject in subjects:
        subject_path = os.path.join(preprocessed_root, subject)
        if not os.path.isdir(subject_path):
            continue

        edf_subject_path = os.path.join(edf_root, subject)
        print(f"\n=== Processing Subject: {subject} ===")
        process_subject(subject_path, edf_subject_path)

if __name__ == '__main__':
    main()
