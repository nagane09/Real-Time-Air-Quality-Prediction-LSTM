import numpy as np

def create_sequences(data, seq_length=24, target_col='co'):
    X, y = [], []
    data_array = data.values
    target_idx = data.columns.get_loc(target_col)
    
    for i in range(len(data_array) - seq_length):
        X.append(data_array[i:i+seq_length])
        y.append(data_array[i+seq_length][target_idx])
        
    return np.array(X), np.array(y)
