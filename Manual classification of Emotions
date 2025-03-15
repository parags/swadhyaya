!pip install neurokit2 mne pandas numpy scikit-learn

import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for clean output

# Set Parameters
duration = 30  # Duration in seconds
fs = 1000  # Sampling frequency (Hz)

# Function to generate synthetic physiological data (for testing)
# Modified to allow specifying target emotion for more control over data generation
def generate_synthetic_data(target_emotion="Neutral"):
    """Generates synthetic physiological data with a bias towards a target emotion."""

    # Base signal generation
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=fs)
    eda_signal = nk.eda_simulate(duration=duration, sampling_rate=fs)
    eeg_alpha_signal = nk.eeg_simulate(duration=duration, sampling_rate=fs)
    eeg_beta_signal = nk.eeg_simulate(duration=duration, sampling_rate=fs)

    # Adjust signals based on target emotion (example adjustments)
    if target_emotion == "Stress":
        # Increase heart rate (decrease MeanNN)
        ecg_signal = nk.signal_resample(ecg_signal, sampling_rate=fs, desired_sampling_rate=fs * 1.2)
        # Increase EDA (phasic and tonic)
        eda_signal = eda_signal + 0.2
    elif target_emotion == "Fear":
        # Increase heart rate variability (SDNN, RMSSD)
        ecg_signal = nk.signal_resample(ecg_signal, sampling_rate=fs, desired_sampling_rate=fs * 1.1)
        # Increase EDA (tonic more than phasic)
        eda_signal = eda_signal + 0.15
    elif target_emotion == "Distress":
        # Decrease heart rate variability (SDNN, RMSSD)
        ecg_signal = nk.signal_resample(ecg_signal, sampling_rate=fs, desired_sampling_rate=fs * 0.9)  # Slightly decrease HR
        # Increase EDA (phasic more than tonic)
        eda_signal = eda_signal + 0.1
    elif target_emotion == "Anxiety":
        # Increase heart rate and variability (SDNN, RMSSD)
        ecg_signal = nk.signal_resample(ecg_signal, sampling_rate=fs, desired_sampling_rate=fs * 1.15)
        # Moderate increase in EDA
        eda_signal = eda_signal + 0.05
        # Adjust EEG (increase beta waves, decrease alpha waves) - optional

    # Ensure signals have the same length
    min_length = min(len(ecg_signal), len(eda_signal), len(eeg_alpha_signal), len(eeg_beta_signal))
    return ecg_signal[:min_length], eda_signal[:min_length], eeg_alpha_signal[:min_length], eeg_beta_signal[:min_length]


# Process a single set of physiological signals
def process_signals(ecg_signal, eda_signal, eeg_alpha_signal, eeg_beta_signal):
    # Process ECG
    ecg_signals, ecg_info = nk.ecg_process(ecg_signal, sampling_rate=fs)

    # Process EDA
    eda_signals, eda_info = nk.eda_process(eda_signal, sampling_rate=fs)

    # EEG Power Analysis
    eeg_alpha_features = nk.eeg_power(eeg_alpha_signal, sampling_rate=fs)
    eeg_beta_features = nk.eeg_power(eeg_beta_signal, sampling_rate=fs)

    # Extract HRV (Heart Rate Variability) Features
    hrv_time_features = nk.hrv_time(ecg_info["ECG_R_Peaks"], sampling_rate=fs)
    hrv_freq_features = nk.hrv_frequency(ecg_info["ECG_R_Peaks"], sampling_rate=fs, psd_method="welch")

    # Calculate EEG Complexity (HFD - Higuchi Fractal Dimension)
    eeg_complexity_features = nk.complexity_hjorth(eeg_alpha_signal)
    hfd_alpha = eeg_complexity_features[0]

    return ecg_signals, eda_signals, eeg_alpha_features, eeg_beta_features, hrv_time_features, hrv_freq_features, hfd_alpha, eda_info

# Emotion Mapping Function
def classify_emotion(HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_pNN50, EDA_Phasic_Mean, EDA_Tonic_Mean, HRV_LF_HF, hfd_alpha):
    """
    Classifies emotions based on physiological signals using world-standard thresholds.
    """

    # 1️⃣ **High Stress / Anxiety (Sympathetic Dominance)**
    # - HRV Low: MeanNN < 750ms, SDNN < 30ms
    # - EDA Increased: Phasic > 0.3 µS, Tonic > 0.25 µS
    if HRV_MeanNN < 750 and HRV_SDNN < 30 or (EDA_Phasic_Mean > 0.3 and EDA_Tonic_Mean > 0.25):
        return "High Stress / Anxiety"

    # 2️⃣ **Fear / Nervousness (Fight or Flight Activation)**
    # - HRV Low: SDNN < 40ms, RMSSD < 25ms
    # - EDA: Tonic > 0.2 µS
    elif HRV_SDNN < 40 and HRV_RMSSD < 25 and EDA_Tonic_Mean > 0.2:
        return "Fear / Nervousness"

    # 3️⃣ **Emotional Distress (Physiological Dysregulation)**
    # - HRV: RMSSD < 35ms, pNN50 < 3%
    # - EDA: Phasic > 0.2 µS
    elif HRV_RMSSD < 35 and HRV_pNN50 < 3 and EDA_Phasic_Mean > 0.2:
        return "Emotional Distress"

    # 4️⃣ **Sympathetic Dominance (Chronic Stress Response)**
    # - HRV LF/HF Ratio > 2.5 (More sympathetic activation)
    # - EEG Complexity: HFD Alpha < 1.4 (Cognitive Impairment)
    elif HRV_LF_HF > 2.5 or hfd_alpha < 1.4:
        return "Sympathetic Dominance (Possible Anxiety)"

    # 5️⃣ **Neutral / Positive State**
    # - HRV Healthy: SDNN > 50ms, RMSSD > 40ms
    # - EDA Low: Phasic < 0.15 µS
    # - EEG Complexity: HFD Alpha > 2.0
    elif HRV_SDNN > 50 and HRV_RMSSD > 40 and EDA_Phasic_Mean < 0.15 and hfd_alpha > 2.0:
        return "Neutral / Positive State"

    # 6️⃣ **Calm & Rested (Parasympathetic Dominance)**
    # - HRV: SDNN > 60ms, RMSSD > 50ms
    # - EDA Low: Tonic < 0.1 µS
    elif HRV_SDNN > 60 and HRV_RMSSD > 50 and EDA_Tonic_Mean < 0.1:
        return "Calm & Rested (Parasympathetic Dominance)"

    else:
        return "Unclassified"


# Main Execution Block
if __name__ == "__main__":

    # Test cases for different emotions
    emotions_to_test = ["Neutral", "Stress", "Fear", "Distress", "Anxiety"]

    for emotion in emotions_to_test:
        print(f"\n---- Testing for {emotion} ----")
        # Generate synthetic data with bias towards the target emotion
        ecg_signal, eda_signal, eeg_alpha_signal, eeg_beta_signal = generate_synthetic_data(target_emotion=emotion)

        # Process the physiological signals
        ecg_signals, eda_signals, eeg_alpha_features, eeg_beta_features, hrv_time, hrv_freq, hfd_alpha, eda_info = process_signals(
            ecg_signal, eda_signal, eeg_alpha_signal, eeg_beta_signal
        )

        # Classify emotion
        # Extract necessary values safely
        HRV_MeanNN = hrv_time.get("HRV_MeanNN", pd.Series([np.nan])).values[0]
        HRV_SDNN = hrv_time.get("HRV_SDNN", pd.Series([np.nan])).values[0]
        HRV_RMSSD = hrv_time.get("HRV_RMSSD", pd.Series([np.nan])).values[0]
        HRV_pNN50 = hrv_time.get("HRV_pNN50", pd.Series([np.nan])).values[0]
        HRV_LF_HF = hrv_freq.get("HRV_LF", pd.Series([np.nan])).values[0] / max(hrv_freq.get("HRV_HF", pd.Series([1])).values[0], 1)

        EDA_Phasic_Mean = eda_info.get("EDA_Phasic_Mean", pd.Series([np.nan])).mean()
        EDA_Tonic_Mean = eda_info.get("EDA_Tonic_Mean", pd.Series([np.nan])).mean()

        detected_emotion = classify_emotion(HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_pNN50, EDA_Phasic_Mean, EDA_Tonic_Mean, HRV_LF_HF, hfd_alpha)

        # Print Results
        print(f"\n🔥 **Detected Emotion: {detected_emotion}**")
