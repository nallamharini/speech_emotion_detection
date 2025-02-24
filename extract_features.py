import os
import librosa
import numpy as np
import pandas as pd

# Set dataset path
dataset_path = r"C:\Users\nalla\OneDrive\Desktop\speech_emotion_detection\SAVEE dataset\ALL"
output_csv = "features.csv"

# Function to extract MFCC features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)  # Load audio
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract MFCCs
        mfccs_mean = np.mean(mfccs, axis=1)  # Compute mean of MFCCs
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Detect WAV files in dataset
files = os.listdir(dataset_path)
wav_files = [f for f in files if f.endswith(".wav")]

# Debugging: Check detected WAV files
if not wav_files:
    print(" No WAV files found in the dataset directory! Check the dataset path.")
else:
    print(f" Found {len(wav_files)} WAV files.")

data = []  # List to store extracted features

# Process each WAV file
for filename in wav_files:
    file_path = os.path.join(dataset_path, filename)
    
    # Extract features
    features = extract_features(file_path)
    
    if features is not None:
        print(f" Extracted {len(features)} features from: {filename}")  # Debugging
        
        # Extract emotion label from filename (Assuming SAVEE naming convention)
        label = filename[0]  # Example: 'a01.wav' -> 'a' (Angry)
        
        data.append(np.append(features, label))  # Append features + label

# Convert to DataFrame
if data:
    columns = [f"feature_{i}" for i in range(40)] + ["label"]  # Column names
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f" Features saved successfully to {output_csv} with {len(df)} entries.")
else:
    print(" No features extracted. Check dataset path and WAV files.")
