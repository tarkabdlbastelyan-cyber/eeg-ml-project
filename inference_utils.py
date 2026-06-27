
import numpy as np
import mne
import joblib
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from collections import Counter

def load_eeg_only(file_path):
    """
    Load EDF file and keep only EEG brain channels.
    Removes ECG and A2-A1 reference channel.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    eeg_channels = [
        ch for ch in raw.ch_names
        if ("ECG" not in ch) and ("A2-A1" not in ch)
    ]

    raw.pick(eeg_channels)

    # Bandpass filtering
    raw.filter(l_freq=0.5, h_freq=30, verbose=False)

    return raw


def crop_to_60_seconds(raw):
    """
    Crop signal to first 60 seconds if longer than 60 seconds.
    In GUI use, we do not know the ground truth label, so we use first 60 seconds.
    """
    duration_sec = 60

    if raw.times[-1] > duration_sec:
        raw.crop(tmin=0, tmax=duration_sec)

    return raw


def segment_raw(raw, window_sec=4, step_sec=2):
    """
    Split EEG signal into 4-second windows with 2-second overlap.
    """
    fs = int(raw.info["sfreq"])
    data = raw.get_data() * 1e6  # Volt to microvolt

    window_size = window_sec * fs
    step_size = step_sec * fs

    segments = []

    for start in range(0, data.shape[1] - window_size + 1, step_size):
        end = start + window_size
        segment = data[:, start:end]
        segments.append(segment)

    return np.array(segments), fs


def extract_features_from_segment(segment, fs):
    """
    Extract DSP features from one EEG segment.
    Features:
    mean, std, RMS, skewness, kurtosis, energy,
    delta power, theta power, alpha power, beta power.
    """
    features = []

    frequency_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30)
    }

    for ch_signal in segment:
        mean_value = np.mean(ch_signal)
        std_value = np.std(ch_signal)
        rms_value = np.sqrt(np.mean(ch_signal ** 2))
        skew_value = skew(ch_signal)
        kurtosis_value = kurtosis(ch_signal)
        energy_value = np.sum(ch_signal ** 2) / len(ch_signal)

        features.extend([
            mean_value,
            std_value,
            rms_value,
            skew_value,
            kurtosis_value,
            energy_value
        ])

        freqs, psd = welch(
            ch_signal,
            fs=fs,
            nperseg=min(1024, len(ch_signal))
        )

        for band_name, band_range in frequency_bands.items():
            band_idx = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])

            if np.sum(band_idx) == 0:
                band_power = 0
            else:
                band_power = np.trapezoid(psd[band_idx], freqs[band_idx])

            features.append(band_power)

    return np.array(features)


def predict_eeg_file(file_path, model_path="best_soft_voting_eeg_model.pkl"):
    """
    Full prediction pipeline for GUI.

    Input:
    EDF file path

    Output:
    dictionary containing:
    - final_class
    - final_label
    - low_segments
    - high_segments
    - high_percentage
    - total_segments
    """

    model = joblib.load(model_path)

    raw = load_eeg_only(file_path)
    raw = crop_to_60_seconds(raw)

    segments, fs = segment_raw(raw)

    X = []
    for segment in segments:
        features = extract_features_from_segment(segment, fs)
        X.append(features)

    X = np.array(X)

    segment_predictions = model.predict(X)

    counts = Counter(segment_predictions)

    low_count = counts.get(0, 0)
    high_count = counts.get(1, 0)

    if high_count > low_count:
        final_class = 1
        final_label = "High Workload / Mental Arithmetic"
    else:
        final_class = 0
        final_label = "Low Workload / Baseline"

    high_percentage = (high_count / len(segment_predictions)) * 100

    return {
        "final_class": final_class,
        "final_label": final_label,
        "low_segments": low_count,
        "high_segments": high_count,
        "high_percentage": high_percentage,
        "total_segments": len(segment_predictions)
    }
