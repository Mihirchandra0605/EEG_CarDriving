import os
import numpy as np
import mne
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

raw = mne.io.read_raw_edf("../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files/S007/S007R12.edf", preload=True)


LOW_CUTOFF = 8  
HIGH_CUTOFF = 30

raw.filter(LOW_CUTOFF, HIGH_CUTOFF, fir_design='firwin')

raw.notch_filter(freqs=50)

ica = ICA(n_components=25, random_state=42)
ica.fit(raw)
raw = ica.apply(raw)

eeg_data = raw.get_data()

pca = PCA(n_components=20)  # 20 components
eeg_pca = pca.fit_transform(eeg_data.T) 

scaler = StandardScaler()
eeg_pca_normalized = scaler.fit_transform(eeg_pca)

raw.plot_psd()
raw.plot(title="EEG Data - after Preprocessing")


plt.show()