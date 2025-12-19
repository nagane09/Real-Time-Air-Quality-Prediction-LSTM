import torch
from model_lstm import LSTMModel
import numpy as np

def predict_next_hour(scaled_sequence, model_path="models/lstm_model.pt"):
    input_size = scaled_sequence.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32)
        prediction = model(input_tensor).numpy()
    return prediction
