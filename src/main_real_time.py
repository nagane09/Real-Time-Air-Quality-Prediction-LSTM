import numpy as np
import torch
import pandas as pd
from src.fetch_api import fetch_live_data
from src.preprocessing import scale_live_data
from src.model_lstm import LSTMModel

df_live = fetch_live_data()
print("Live Data:")
print(df_live)

scaled = scale_live_data(df_live)


sequence = np.tile(scaled, (1,24,1))
sequence = torch.tensor(sequence, dtype=torch.float32)

input_size = scaled.shape[1]
model = LSTMModel(input_size=input_size)
model.load_state_dict(torch.load('models/lstm_model.pt', map_location='cpu'))
model.eval()

with torch.no_grad():
    prediction = model(sequence)
predicted_co = prediction[0][0].item()
print(f"Predicted CO next hour: {predicted_co:.2f}")
