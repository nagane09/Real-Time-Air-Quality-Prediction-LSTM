
import requests
import pandas as pd
from datetime import datetime

API_TOKEN = "b75cfc80984062971fcf149d376415677cdedfb7"
API_URL = f"https://api.waqi.info/feed/delhi/?token={API_TOKEN}"

LSTM_FEATURES = ['co', 'no2', 'o3']

API_MAP = {
    'co': 'co',
    'no2': 'no2',
    'o3': 'o3'
}

def fetch_live_data():
    response = requests.get(API_URL).json()
    
    if response['status'] != 'ok':
        raise ValueError(f"API returned status: {response['status']}")
    
    iaqi = response['data'].get('iaqi', {})
    
    record = {}
    record['datetime'] = datetime.now()
    record['co'] = iaqi.get('co', {}).get('v', 0)
    record['no2'] = iaqi.get('no2', {}).get('v', 0)
    record['o3'] = iaqi.get('o3', {}).get('v', 0)
    
    df = pd.DataFrame([record])
    return df
