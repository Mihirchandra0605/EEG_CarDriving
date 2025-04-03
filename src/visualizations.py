import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

def load_single_run(subject, run, raw_data_dir, preprocessed_data_dir, selected_channels=10):
    """
    Load EEG data for a single subject and a specific run.
    
    Parameters:
    - subject (int): Subject number (e.g., 1 for S001)
    - run (int): Run number (e.g., 3 for S001R03)
    - raw_data_dir (str): Path to raw EEG data folder
    - preprocessed_data_dir (str): Path to preprocessed EEG data folder
    - selected_channels (int or list): Number of channels to select or a list of specific channels

    Returns:
    - raw_data (np.ndarray): Raw EEG data (time x channels)
    - clean_data (np.ndarray): Preprocessed EEG data (time x channels)
    - channel_names (list): Names of selected EEG channels
    - times (np.ndarray): Time array for x-axis
    """
    # Format subject and run IDs correctly
    subject_raw = f"S{subject:03d}"
    subject_preprocessed = f"S{subject}"
    run_id = f"R{run:02d}"

    # Paths for raw and preprocessed data
    raw_file = os.path.join(raw_data_dir, subject_raw, f"{subject_raw}{run_id}.edf")
    preprocessed_file = os.path.join(preprocessed_data_dir, subject_preprocessed, f"{subject_raw}{run_id}_preprocessed.npy")

    # Load raw data
    raw = mne.io.read_raw_edf(raw_file, preload=True)
    raw_data = raw.get_data()
    channel_names = raw.ch_names
    times = np.linspace(0, raw.times[-1], raw_data.shape[1])  # Time axis

    # Load preprocessed data
    clean_data = np.load(preprocessed_file)

    # Ensure selected channels is a list
    if isinstance(selected_channels, int):
        selected_channels = list(range(selected_channels))  # First N channels

    # Select only the required channels
    raw_data = raw_data[selected_channels, :].T
    clean_data = clean_data[selected_channels, :].T
    channel_names = [channel_names[i] for i in selected_channels]

    return raw_data, clean_data, channel_names, times

def plot_single_run(subject, run, raw_data_dir, preprocessed_data_dir, selected_channels=10):
    """
    Plot EEG signals for a single subject and a specific run.
    
    Parameters:
    - subject (int): Subject number (e.g., 1 for S001)
    - run (int): Run number (e.g., 3 for S001R03)
    - raw_data_dir (str): Path to raw EEG data folder
    - preprocessed_data_dir (str): Path to preprocessed EEG data folder
    - selected_channels (int or list): Number of channels to select or a list of specific channels
    """
    raw_data, clean_data, channel_names, times = load_single_run(subject, run, raw_data_dir, preprocessed_data_dir, selected_channels)

    print("Times shape:", times.shape)  # Expected: (20000,)
    print("Raw Data shape:", raw_data.shape)  # Expected: (20000, num_channels)
    print("Clean Data shape:", clean_data.shape)  # Expected: (20000, num_channels)


    # Normalize the data for better visualization
    raw_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
    clean_data = (clean_data - np.mean(clean_data, axis=0)) / np.std(clean_data, axis=0)
    times_clean = np.linspace(times[0], times[-1], clean_data.shape[0])  # Rescale time axis


    # Plot
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(12, len(channel_names) * 2), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(times, raw_data[:, i], label="Raw Data", color="red", alpha=0.6)
        ax.plot(times_clean, clean_data[:, i], label="Preprocessed Data", color="blue", alpha=0.6)
        ax.set_ylabel(channel_names[i])
        ax.legend(loc="upper right")
        ax.grid(True)

    plt.xlabel("Time (seconds)")
    plt.suptitle(f"EEG Comparison for Subject {subject:03d}, Run {run:02d}")
    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    raw_data_dir = "../data/eeg-motor-movementimagery-dataset-1.0.0 (1)/files"
    preprocessed_data_dir = "../data/sorted_data"

    # Choose subject, run, and number of channels
    plot_single_run(subject=77, run=14, raw_data_dir=raw_data_dir, preprocessed_data_dir=preprocessed_data_dir, selected_channels=10)
