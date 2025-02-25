import os
import numpy as np
import pandas as pd
import librosa

dataset_path = r"C:\Users\nalla\OneDrive\Desktop\speech_emotion_detection\SAVEE dataset\ALL"
output_csv = "features.csv"

# 🔹 Extract Features Function
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)  # Load audio file
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=153)  # Increase MFCC to 153
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)  # Chroma
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)  # Mel Spectrogram
        
        #  Convert to 1D array (flatten) and combine features
        features = np.hstack([ 
            np.mean(mfccs, axis=1), 
            np.mean(chroma, axis=1), 
            np.mean(mel, axis=1)
        ])
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# 🔹 Initialize Lists
data = []
labels = []

# 🔹 Process All WAV Files
for filename in os.listdir(dataset_path):
    if filename.endswith(".wav"):  # Ensure only .wav files are processed
        file_path = os.path.join(dataset_path, filename)
        
        # Extract features
        features = extract_features(file_path)

        if features is not None:
            data.append(features)
            
            # Extract label from filename (SAVEE format)
            label = filename[0]  # Modify this if filenames have different emotion codes
            labels.append(label)
            
            print(f"Processed: {filename} - Label: {label}")

# 🔹 Convert to DataFrame and Save
if len(data) > 0:
    feature_columns = [f"feature_{i}" for i in range(len(data[0]))]  # Feature names
    df = pd.DataFrame(data, columns=feature_columns)
    df["label"] = labels  # Add label column
    
    print(df.head())  # Show first few rows for debugging
    
    df.to_csv(output_csv, index=False)  # Save to CSV
    print(f"Features saved to {output_csv}")
else:
    print("No features extracted. Check dataset path and WAV files.")
