import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from decoder import decode_features

# In train_model.py
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

def load_model():
    features, labels = decode_features('features.csv')
    model, le = train_lstm_model(features, labels)
    return model, le

if __name__ == "__main__":
    model, label_encoder = load_model()
    model.save('emotion_model.h5')  # Model is already compiled
    np.save('label_encoder.npy', label_encoder.classes_)