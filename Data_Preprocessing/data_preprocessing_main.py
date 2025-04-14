import os
import numpy as np
import mne
from mne.preprocessing import ICA

# === TASK MAP ===
RUN_LABEL_MAP = {
    'T1': {
        (3, 4, 7, 8, 11, 12): 'left_fist',
        (5, 6, 9, 10, 13, 14): 'both_fists'
    },
    'T2': {
        (3, 4, 7, 8, 11, 12): 'right_fist',
        (5, 6, 9, 10, 13, 14): 'both_feet'
    },
    'T0': 'rest'
}

# === CHANNEL SELECTION ===
MOTOR_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.'
]

def get_true_label(run_num, annot_code):
    if annot_code == 'T0':
        return RUN_LABEL_MAP['T0']
    for run_set, label in RUN_LABEL_MAP.get(annot_code, {}).items():
        if run_num in run_set:
            return label
    return 'unknown'

def preprocess_and_segment_by_annotation(edf_dir, save_dir):
    subjects = sorted(os.listdir(edf_dir))

    for subject in subjects:
        subject_path = os.path.join(edf_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        print(f"\n Processing Subject: {subject}")
        for edf_file in sorted(os.listdir(subject_path)):
            if not edf_file.endswith('.edf'):
                continue

            try:
                run_num = int(edf_file.split('R')[-1].split('.')[0])
                edf_path = os.path.join(subject_path, edf_file)
                print(f"  EDF: {edf_file} | Run: {run_num}")

                raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)

                # Select only motor-related channels
                raw.pick_channels([ch for ch in MOTOR_CHANNELS if ch in raw.ch_names])

                # === Preprocessing ===
                raw.filter(8., 30., fir_design='firwin', verbose=False)
                raw.notch_filter(freqs=50., verbose=False)
                raw.set_eeg_reference('average', projection=True)

                # ICA
                ica = ICA(n_components=15, random_state=97, max_iter='auto')
                ica.fit(raw)
                raw = ica.apply(raw)

                annotations = raw.annotations
                if len(annotations) == 0:
                    print("   No annotations found. Skipping...")
                    continue

                subject_save_path = os.path.join(save_dir, subject)
                os.makedirs(subject_save_path, exist_ok=True)

                for i in range(len(annotations) - 1):
                    onset = annotations[i]['onset']
                    offset = annotations[i + 1]['onset']
                    label_code = annotations[i]['description']

                    if label_code not in ['T0', 'T1', 'T2']:
                        continue

                    label = get_true_label(run_num, label_code)
                    segment = raw.copy().crop(tmin=onset, tmax=offset, include_tmax=False)

                    # Save segment
                    data = segment.get_data()
                    filename = f"run_{run_num:02d}_seg_{i:03d}_{label}.npy"
                    np.save(os.path.join(subject_save_path, filename), data)

                print(f" Saved segments to {subject_save_path}")

            except Exception as e:
                print(f" Error processing {edf_file}: {e}")
                continue

if __name__ == '__main__':
    EDF_ROOT = '../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files'
    SAVE_ROOT = '../data/Preprocessed_data_main2'
    preprocess_and_segment_by_annotation(EDF_ROOT, SAVE_ROOT)
