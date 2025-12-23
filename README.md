# 🌍 Real-Time Air Quality Prediction Using LSTM (Deep Learning)

🔴 **Live Demo:**  
👉 https://nagane09-real-time-air-quality-prediction-lstm-app-qrm2qk.streamlit.app/

---


## 📊 Dataset

* **Source:** Delhi Air Quality CSV
* **Columns:**  
  - `CO Mean`, `NO2 Mean`, `O3 Mean` (main features)
  - `CO 1st Max Value`, `NO2 1st Max Value`, `O3 1st Max Value`, etc. (auxiliary features)
* **Size:** ~1.7 million rows
* **Target variable:** CO concentration (`CO Mean`)

The raw dataset contains missing values and differing units across pollutants, which are handled during preprocessing.

---

## ⚙️ Data Preprocessing

**Steps involved:**

1. **Select Relevant Features:**  
   Columns `CO Mean`, `NO2 Mean`, `O3 Mean` mapped to `co`, `no2`, `o3`.

2. **Handle Missing Values:**  
   Median imputation for any missing values.

3. **Scaling:**  
   MinMaxScaler used to scale features between 0 and 1. Scaler saved as `models/scaler.save` for live predictions.

4. **Sequence Generation:**  
   Time-series sequences of length 24 hours created for LSTM input using `create_sequences` function. Each sequence contains past 24 hours of readings.

```python
X, y = create_sequences(df_scaled, seq_length=24, target_col='co')
`````
# 🌫️ Delhi Air Quality Prediction - LSTM Pipeline

This project predicts **next-hour air pollutant concentrations** (CO, NO2, O3) for Delhi using an **LSTM-based deep learning model**. It includes **data ingestion, preprocessing, sequence generation, model training, real-time prediction, and deployment via Streamlit**.

---

## 🧠 Model Architecture

**Model Type:** LSTM (Long Short-Term Memory) using PyTorch

**Technical Details:**

- **Input Size:** Number of features (3: CO, NO2, O3)
- **Hidden Size:** 32 (configurable)
- **Layers:** 1 LSTM layer (can increase for complexity)
- **Output:** 1 (predict next-hour CO)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam, learning rate = 0.001
- **Batch Size:** 16
- **Epochs:** 10 (can be tuned)

---

## 🧠 What is LSTM?

**LSTM (Long Short-Term Memory)** is a type of **Recurrent Neural Network (RNN)** designed to model **sequential data** while overcoming traditional RNN limitations, like the **vanishing gradient problem**.

### **Key Concepts**

1. **Sequential Data Handling**
   * LSTM is ideal for **time-series data** where current outputs depend on past inputs.
   * Example: Air quality prediction, stock prices, language modeling.

2. **Memory Cell**
   * LSTM contains a **cell state** (`C_t`) that acts like long-term memory.
   * The cell state can **retain information over long sequences**.

3. **Gates in LSTM**  
   LSTM uses three main gates to control information flow:

   * **Forget Gate (`f_t`)**  
     Decides what information from the previous cell state to **discard**.  
     `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`

   * **Input Gate (`i_t`)**  
     Determines which new information to **store** in the cell state.  
     `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`

   * **Output Gate (`o_t`)**  
     Determines what part of the cell state to **output**.  
     `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`

   * **Candidate State (`C̃_t`)**  
     New candidate values that could be added to the cell state.  
     `C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)`

   * **Cell State Update**  
     `C_t = f_t * C_{t-1} + i_t * C̃_t`

   * **Hidden State (`h_t`)**  
     `h_t = o_t * tanh(C_t)`

4. **Advantages of LSTM**
   * Can **remember long-term dependencies** better than vanilla RNNs.
   * Handles **time-series predictions** efficiently.
   * Reduces **vanishing and exploding gradient issues**.

---

### **LSTM in Your Project**

* **Input:** Features like `CO`, `NO2`, `O3`.
* **Sequence Length:** 24 hours (sliding window).
* **Output:** Next-hour CO prediction.
* **Framework:** PyTorch
* **Why LSTM:** Air pollution has **temporal correlations**, so LSTM can capture trends across time to predict the next-hour concentration accurately.

----

## 🏋️ Training Pipeline

**Steps:**

1. Load processed CSV.
2. Generate sequences using `create_sequences`.
3. Split into Train (70%), Validation (15%), Test (15%).
4. Convert sequences to PyTorch tensors.
5. Initialize LSTM model.
6. Train with MSE loss and Adam optimizer.
7. Save trained model weights as `lstm_model.pt`.

---

## 🌐 Real-Time Prediction

**Steps:**

1. Fetch live data from WAQI API using `fetch_live_data`.
2. Scale live features using saved scaler.
3. Create sequence tensor for LSTM input.
4. Load trained model (`lstm_model.pt`) and predict next-hour CO concentration.

```python
df_live = fetch_live_data()
scaled = scale_live_data(df_live)
sequence = np.tile(scaled, (1, 24, 1))
sequence = torch.tensor(sequence, dtype=torch.float32)

model = LSTMModel(input_size=scaled.shape[1])
model.load_state_dict(torch.load("models/lstm_model.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    prediction = model(sequence)
````



## 📈 Visualization & Alerts

- Use **matplotlib** to plot historical vs predicted values.  
- Optionally, send **email alerts** if predicted PM2.5 or CO exceeds safe thresholds using `smtplib`.

---

## 🖥️ Streamlit Deployment

**File:** `app.py`  
**Functionality:**

- Fetch live data
- Preprocess
- Predict next-hour CO
- Display results in a dashboard

```bash
streamlit run app.py
````

## 🔧 Technologies & Libraries

| Component       | Libraries/Tools                   |
|-----------------|----------------------------------|
| Data Handling   | pandas, numpy                     |
| Preprocessing   | scikit-learn (MinMaxScaler), joblib |
| Deep Learning   | PyTorch                            |
| API Fetch       | requests, datetime                |
| Scheduling      | schedule, time                    |
| Visualization   | matplotlib, Streamlit             |
| Email Alerts    | smtplib, email.mime               |

---

## 📊 Impact & Sustainability

- **Reduce Health Risks:** Predicting pollutants helps authorities issue warnings and mitigate exposure.  
- **Data-Driven Insights:** Identify trends in Delhi air quality to inform urban planning and policy.  
- **Transparency:** Real-time dashboard allows citizens to see predictions and trends.

---

## 🛡️ Security & Compliance

- Sensitive API tokens and credentials are stored securely.  
- All user interactions with the dashboard are read-only; no sensitive data is exposed.  
- Code follows standard Python and PyTorch best practices for maintainability.

