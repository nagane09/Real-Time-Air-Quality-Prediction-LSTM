import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

CSV_FEATURES = ['CO Mean', 'NO2 Mean', 'O3 Mean']
LSTM_FEATURES = ['co', 'no2', 'o3']
COLUMN_MAP = {
    'CO Mean': 'co',
    'NO2 Mean': 'no2',
    'O3 Mean': 'o3'
}

def preprocess_air_quality(df, save_scaler_path='../models/scaler.save'):
    available_cols = [col for col in CSV_FEATURES if col in df.columns]
    df = df[available_cols]
    
    df = df.rename(columns=COLUMN_MAP)
    
    df[LSTM_FEATURES] = df[LSTM_FEATURES].fillna(df[LSTM_FEATURES].median())
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[LSTM_FEATURES] = scaler.fit_transform(df[LSTM_FEATURES])
    
    joblib.dump(scaler, save_scaler_path)
    
    return df_scaled


def scale_live_data(df, scaler_path="models/scaler.save"):
    scaler = joblib.load(scaler_path)
    
    for feat in LSTM_FEATURES:
        if feat not in df.columns:
            df[feat] = 0
    
    df_features = df[LSTM_FEATURES].copy()
    df_features.fillna(0, inplace=True)
    
    scaled = scaler.transform(df_features)
    return scaled
