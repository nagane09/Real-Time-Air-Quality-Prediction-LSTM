import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_lstm import LSTMModel
from utils import create_sequences

# 1. Load processed CSV
df_scaled = pd.read_csv('data/processed/Delhi_Air_Quality_Processed.csv')

# 2. Create sequences (use 'co' as target for example)
X, y = create_sequences(df_scaled, seq_length=24, target_col='co')

# 3. Train-test split
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# 4. Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 5. Initialize model
input_size = X_train.shape[2]  # number of features (3)
model = LSTMModel(input_size=input_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# 7. Save model
torch.save(model.state_dict(), 'models/lstm_model.pt')
