import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from flask import Flask, request, jsonify
import glob

# Flask app
app = Flask(__name__)

# Feature extraction function (optimized)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Simplified pitch (using zero-crossing rate instead of piptrack for speed)
        pitch_proxy = librosa.feature.zero_crossing_rate(y)
        pitch_mean = np.mean(pitch_proxy)
        pitch_std = np.std(pitch_proxy)
        
        # Frequency-related features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Energy-related features
        rms = np.mean(librosa.feature.rms(y=y))
        energy_mean = np.mean(np.abs(y))
        energy_std = np.std(np.abs(y))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'rms_energy': rms,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            **{f'mfcc_{i+1}_mean': mfcc_mean[i] for i in range(len(mfcc_mean))}
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Decoder function to process features.csv
def decode_features(csv_path):
    df = pd.read_csv(csv_path)
    features = df[['pitch_mean', 'pitch_std', 'spectral_centroid', 'spectral_bandwidth', 
                   'rms_energy', 'energy_mean', 'energy_std'] + 
                  [f'mfcc_{i+1}_mean' for i in range(13)]].values
    labels = df['emotion'].values
    return features, labels

# Prepare dataset from RAVDESS
def prepare_dataset(audio_dir):
    features_list = []
    labels = []
    
    files = glob.glob(f"{audio_dir}/Actor_*/*.wav")
    print(f"Found {len(files)} audio files in {audio_dir}")
    print("First 5 files:", files[:5])
    if not files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}/Actor_*/*.wav")
    
    for i, file in enumerate(files):
        emotion_code = int(os.path.basename(file).split('-')[2])
        emotion_map = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
                       5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
        emotion = emotion_map.get(emotion_code, 'unknown')
        if emotion == 'unknown': continue
        
        feature_dict = extract_features(file)
        if feature_dict is None:
            continue
        feature_dict['emotion'] = emotion
        features_list.append(feature_dict)
        labels.append(emotion)
        
        # Progress update every 100 files
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files)} files")
    
    if not features_list:
        raise ValueError("No features extracted from audio files")
    
    df = pd.DataFrame(features_list)
    df.to_csv('features.csv', index=False)
    print(f"Saved {len(df)} samples to features.csv")
    return df

# Train LSTM model
def train_lstm_model(features, labels):
    le = LabelEncoder()
    labels_encoded = to_categorical(le.fit_transform(labels))
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(labels_encoded.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    return model, le

# Load and train model
audio_dir = r'C:\Users\nalla\OneDrive\Desktop\speech_emotion_detection-1\audio\archive (1)'  # Updated path

print("Generating features.csv...")
prepare_dataset(audio_dir)  # Always run to ensure fresh data
features, labels = decode_features('features.csv')
model, label_encoder = train_lstm_model(features, labels)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['audio']
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)
    
    feature_dict = extract_features(audio_path)
    if feature_dict is None:
        return jsonify({'error': 'Failed to extract features from audio'}), 400
    feature_vector = np.array([list(feature_dict.values())])
    feature_vector = feature_vector.reshape((1, 1, feature_vector.shape[1]))
    
    prediction = model.predict(feature_vector)
    emotion_idx = np.argmax(prediction)
    emotion = label_encoder.inverse_transform([emotion_idx])[0]
    
    os.remove(audio_path)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True, port=5000)