import mne
import numpy as np
import matplotlib.pyplot as plt

data = np.load('../data/Preprocessed_data1/S001/S001R03_preprocessed.npy')  # shape: (n_channels, n_samples)
data = data.T 


raw = mne.io.read_raw_edf("../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files/S001/S001R03.edf", preload=True)
events, event_id = mne.events_from_annotations(raw)

info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=160, ch_types='eeg')
raw_array = mne.io.RawArray(data, info)


raw_array.set_annotations(raw.annotations)
events, event_id = mne.events_from_annotations(raw_array)


epochs = mne.Epochs(raw_array, events, event_id=event_id, tmin=0, tmax=2, baseline=None, preload=True)
features = epochs.get_data()
labels = epochs.events[:, -1]
print(labels)