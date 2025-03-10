from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from extract_features import extract_features
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8000"}})  # Allow only port 8000

# Load pre-trained model and label encoder
model = keras_load_model('emotion_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy', allow_pickle=True)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received predict request")
    audio_file = request.files['audio']
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)
    print(f"Saved audio to {audio_path}")
    
    feature_dict = extract_features(audio_path)
    if feature_dict is None:
        os.remove(audio_path)
        print("Feature extraction failed")
        return jsonify({'error': 'Failed to extract features from audio'}), 400
    
    print("Features extracted:", feature_dict)
    feature_dict.pop('emotion', None)  # Remove 'emotion' if present
    feature_vector = [list(feature_dict.values())]
    feature_vector = np.array(feature_vector)
    expected_features = 20
    if feature_vector.shape[1] != expected_features:
        os.remove(audio_path)
        print(f"Feature mismatch: expected {expected_features}, got {feature_vector.shape[1]}")
        return jsonify({'error': f'Feature mismatch: expected {expected_features}, got {feature_vector.shape[1]}'}), 400
    
    feature_vector = feature_vector.reshape((1, 1, feature_vector.shape[1]))
    prediction = model.predict(feature_vector)
    emotion_idx = np.argmax(prediction)
    emotion = label_encoder.inverse_transform([emotion_idx])[0]
    print(f"Predicted emotion: {emotion}")
    
    os.remove(audio_path)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True, port=5000)