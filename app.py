import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load trained model, scaler, and label encoder
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define label mapping (adjust as per your dataset labels)
label_map = {
    "A": "Angry",
    "H": "Happy",
    "S": "Sad",
    "N": "Neutral",
    "D": "Disgust",
    "F": "Fear",
    "U": "Surprise"
}

# Define upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_features(audio_path):
    """ Extracts features from an audio file for emotion classification. """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)  # Take mean of MFCCs
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    """ Handles audio file uploads and returns emotion prediction. """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Extract features
    features = extract_features(file_path)
    if features is None:
        return jsonify({"error": "Could not extract features"}), 500

    # Normalize features
    features = scaler.transform([features])

    # Predict emotion
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    predicted_emotion = label_map.get(predicted_label, "Unknown")

    return jsonify({"emotion": predicted_emotion})

if __name__ == "__main__":
    app.run(debug=True)
