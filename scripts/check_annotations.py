import mne

# Load the EDF file
raw = mne.io.read_raw_edf('../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files/S106/S106R05.edf', preload=True)
annotations = raw.annotations

# You can see all the onsets, durations, and descriptions (like T1, T2)
for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
    print(f"Time: {onset:.2f}s, Duration: {duration:.2f}s, Label: {description}")

