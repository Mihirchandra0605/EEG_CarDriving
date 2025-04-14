import mne

# Load the raw EEG data
raw = mne.io.read_raw_edf('../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files/S001/S001R01.edf', preload=True)

# Print the channel names
print(raw.ch_names)

# Plot the EEG signal
raw.plot(n_channels=10, duration=5, scalings='auto', title='EEG Raw Data', show=True, block=True)
