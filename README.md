# Real-Time-Air-Quality-Prediction-LSTM

# 🌍 Real-Time Air Quality Prediction Using LSTM (Deep Learning)

🔴 **Live Demo:**  
👉 https://nagane09-real-time-air-quality-prediction-lstm-app-qrm2qk.streamlit.app/

---

## 1. Project Overview

This project implements a **real-time air quality prediction system** using a **Long Short-Term Memory (LSTM)** deep learning model.  
The system forecasts **future pollutant concentration (next-hour prediction)** using historical air quality data combined with **live sensor data** fetched from a public API.

The project is designed to be:
- **CPU-friendly**
- **Real-time**
- **Research-ready**
- **Deployable as a web application**

---

## 2. What Problem Does This Project Solve?

Most air quality platforms only show **current or past pollution values**.  
This project goes one step further by **predicting future pollution levels**, allowing:

- Early health warnings
- Smart city monitoring
- Environmental decision support
- Preventive public safety measures

---

## 3. What Does the User See?

When a user opens the **Streamlit dashboard**, they see:

- Live air pollutant values (CO, NO₂, O₃)
- Predicted pollutant level for the **next hour**
- Timestamp of the prediction
- Trend direction (rising / falling)
- Alert messages (safe / moderate / hazardous)

### Example Insight
> “CO level is predicted to increase in the next hour – health risk may rise.”

---

## 4. How to Evaluate Results from the Dashboard

### Real-Time Interpretation
- **Current Value:** Latest measurement from live API
- **Predicted Value:** LSTM output for next hour
- **Trend Analysis:** Compare predicted vs current value
- **Alerts:** Triggered when thresholds are crossed

### Offline Model Evaluation
- **Loss Function:** Mean Squared Error (MSE)
- **Lower loss = better prediction**
- Validation performed on unseen historical data

---


### Why LSTM?
- Handles time dependencies
- Suitable for sequential environmental data
- Efficient and lightweight for CPU execution

---

## 6. Dataset Used

**Delhi Air Quality Dataset**
- Hourly pollutant measurements
- Features include CO, NO₂, O₃, etc.
- Small and CPU-friendly

---

## 7. Project Structure and File Responsibilities


### data/
| Path | Description |
|----|-------------|
| `data/raw/` | Original dataset files |
| `data/processed/` | Cleaned and scaled data |

---

### src/
| File | Description |
|----|-------------|
| `fetch_api.py` | Fetches live air quality data from WAQI API |
| `preprocessing.py` | Cleans, scales, and prepares data |
| `model_lstm.py` | Defines the LSTM neural network |
| `train.py` | Trains the LSTM model |
| `evaluate.py` | Evaluates model performance |
| `main_real_time.py` | End-to-end real-time prediction pipeline |
| `utils.py` | Helper utilities for sequence generation |

---

### models/
| File | Purpose |
|----|--------|
| `lstm_model.pt` | Trained LSTM model weights |
| `scaler.save` | Saved MinMaxScaler for real-time data |

---

### streamlit_app.py
- Web-based dashboard
- Displays live data and predictions
- Handles alerts and visualization

---

## 8. Training Details

- **Model:** LSTM
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Epochs:** 10
- **Batch Size:** 16
- **Device:** CPU

---

## 9. Technical Stack

### Programming
- Python 3.10
- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Visualization & Deployment
- Streamlit
- Matplotlib

### APIs & Tools
- World Air Quality Index (WAQI) API
- Joblib



### LSTM Neural Network

