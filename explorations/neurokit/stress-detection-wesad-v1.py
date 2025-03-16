# 🚀 Import Necessary Libraries
import os
import pickle
import pandas as pd
import numpy as np
import neurokit2 as nk
from google.colab import drive
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 🚀 Mount Google Drive
drive.mount('/content/drive')

# Set dataset path (Modify this based on your Google Drive structure)
dataset_path = "/content/drive/My Drive/EMOTION_DETECTION_MODELS/BioSPPy NeuroKit2/WESAD/WESAD/"

# 🚀 Function to Load a Participant's Data
def load_wesad_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data

# 🚀 Function to Extract Features Using NeuroKit2
def extract_features(ecg_signal, eda_signal, sampling_rate=700):
    # Process ECG & Extract HRV Features
    ecg_signals, ecg_info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    r_peaks = ecg_info["ECG_R_Peaks"]
    hrv_features = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)

    # Process EDA & Extract Features
    eda_signals, _ = nk.eda_process(eda_signal, sampling_rate=sampling_rate)  # Ignore eda_info
    
    # Calculate EDA features from eda_signals DataFrame
    eda_phasic_mean = eda_signals['EDA_Phasic'].mean()
    eda_tonic_mean = eda_signals['EDA_Tonic'].mean()
    
    eda_features = pd.DataFrame({
        "EDA_Phasic_Mean": [eda_phasic_mean],
        "EDA_Tonic_Mean": [eda_tonic_mean]
    })

    # Combine Features
    features = pd.concat([hrv_features, eda_features], axis=1)
    return features

# 🚀 Function to Process All Subjects' Data
def process_wesad_dataset(dataset_path):
    subjects = [f for f in os.listdir(dataset_path) if f.startswith('S')]  # List subject folders
    all_features = []

    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject, f"{subject}.pkl")

        if os.path.exists(subject_path):
            print(f"Processing {subject}...")
            data = load_wesad_data(subject_path)

            # Extract signals
            ecg_signal = data["signal"]["chest"]["ECG"].flatten()
            eda_signal = data["signal"]["chest"]["EDA"].flatten()
            labels = data["label"]

            # Extract features
            features = extract_features(ecg_signal, eda_signal)
            features["Label"] = labels[:len(features)]  # Assign labels

            all_features.append(features)

    # Combine all subjects' data
    full_dataset = pd.concat(all_features, ignore_index=True)
    return full_dataset

# 🚀 Process WESAD Dataset
wesad_data = process_wesad_dataset(dataset_path)

# 🚀 Prepare Data for Machine Learning
X = wesad_data.drop(columns=["Label"])
y = wesad_data["Label"].apply(lambda x: 1 if x == 2 else 0)  # Convert labels (1 = Stress, 0 = No Stress)

# 🚀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🚀 Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# 🚀 Display Results
print("✅ Model Training Complete!")
print(f"🎯 Model Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:\n", classification_rep)
