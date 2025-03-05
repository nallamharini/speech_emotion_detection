import pandas as pd
import numpy as np

def decode_features(csv_path):
    df = pd.read_csv(csv_path)
    features = df[['pitch_mean', 'pitch_std', 'spectral_centroid', 'spectral_bandwidth', 
                   'rms_energy', 'energy_mean', 'energy_std'] + 
                  [f'mfcc_{i+1}_mean' for i in range(13)]].values
    labels = df['emotion'].values
    return features, labels