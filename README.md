# 🌦 India Weather Forecast — LSTM

A deep learning project that predicts the next **7 days of temperature** (max & min) for 10 major Indian cities using a multi-variate LSTM built from scratch in **PyTorch**.

---

## 📌 Project Overview

| | |
|---|---|
| **Dataset** | India Daily Weather 2000–2024 (10 major cities) |
| **Model** | Multi-layer LSTM (built from scratch in PyTorch) |
| **Input** | Last 30 days × 10 weather features |
| **Output** | Next 7 days of Tmax & Tmin (°C) |
| **Interface** | Flask web app with interactive Chart.js graph |

---

## 🏙️ Supported Cities

Delhi · Mumbai · Kolkata · Chennai · Bangalore · Hyderabad · Ahmedabad · Pune · Jaipur · Lucknow

---

## 📁 Project Structure

```
weather-lstm/
│
├── data/
│   └── india_weather.csv           # Raw dataset (2000–2024)
│
├── src/
│   ├── utils.py                    # Data loading, scaling, windowing
│   ├── dataset.py                  # PyTorch Dataset (sliding window)
│   ├── model.py                    # LSTM architecture from scratch
│   ├── train.py                    # Training loop
│   └── evaluate.py                 # Inference + inverse transform
│
├── checkpoints/
│   ├── lstm_weather.pt             # Saved model weights
│   └── scalers/                    # Per-city StandardScaler (.pkl files)
│
├── app/
│   ├── app.py                      # Flask application
│   └── templates/
│       └── index.html              # Frontend with Chart.js
│
└── requirements.txt
```

---

## 🧠 Model Architecture

```
Input: (batch, 30, 10)
        ↓
  2-Layer Stacked LSTM  [hidden_size=128, dropout=0.2]
        ↓
  Last Hidden State  (batch, 128)
        ↓
  Fully Connected Layer  (128 → 14)
        ↓
  Reshape  (batch, 7, 2)
        ↓
Output: 7 days × [Tmax, Tmin]
```

### Why LSTM?

Vanilla RNNs suffer from **vanishing gradients** — they forget events from 20+ days ago. LSTM solves this with a **cell state** (long-term memory highway) controlled by 3 gates:

- **Forget gate** — what fraction of old memory to keep
- **Input gate** — how much new information to write
- **Output gate** — what part of memory to expose as hidden state

This lets the model "remember" that monsoon season started 3 weeks ago when predicting today's temperature.

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `temperature_2m_max` | Daily max temperature (°C) ← **predicted** |
| `temperature_2m_min` | Daily min temperature (°C) ← **predicted** |
| `apparent_temperature_max` | Feels-like max temp |
| `apparent_temperature_min` | Feels-like min temp |
| `precipitation_sum` | Total daily precipitation (mm) |
| `rain_sum` | Total daily rainfall (mm) |
| `weather_code` | WMO weather condition code |
| `wind_speed_10m_max` | Max wind speed at 10m (km/h) |
| `wind_gusts_10m_max` | Max wind gusts at 10m (km/h) |
| `wind_direction_10m_dominant` | Dominant wind direction (°) |

---

## ⚙️ Key Design Decisions

**One model for all cities**
Instead of 10 separate models, a single shared model learns universal weather dynamics. The city context is implicitly captured via per-city normalization (each city gets its own `StandardScaler`). This means more training data per model and better generalization.

**Per-city StandardScaler**
Mumbai's monsoon rainfall is completely different from Jaipur's desert climate. Fitting a scaler per city ensures each city's features are normalized relative to its own historical distribution — not polluted by other cities' ranges.

**Sliding window dataset**
Each training sample is a `(30-day input → 7-day target)` pair. Sliding this window by 1 day across 25 years × 10 cities gives ~90,000 training samples.

**Inverse transform only temperature columns**
The model predicts in scaled space. To convert back to °C, we use a dummy array trick — plug predictions into the correct columns of a zero array, inverse transform all 10 columns, then extract just the 2 temperature columns.

---

## 🚀 Setup & Running

### 1. Install dependencies
```bash
pip install torch flask pandas numpy scikit-learn
```

### 2. Train the model
```bash
cd src
py train.py
```

Training output:
```
Training on: cpu
Loading data...
Total samples: 90960
Train batches: 1280 | Val batches: 143
Epoch [01/50] Train Loss: x.xxxx | Val Loss: x.xxxx
  ✓ Saved best model (val_loss: x.xxxx)
...
Training complete. Best val loss: 0.0480
```

### 3. Generate scalers (if missing)
```bash
cd src
py -c "
from utils import load_data, fit_scalers
df = load_data(r'path\to\data\india_weather.csv')
fit_scalers(df, save_dir=r'path\to\checkpoints\scalers')
print('Scalers saved!')
"
```

### 4. Launch Flask app
```bash
cd app
py app.py
```

### 5. Open browser
```
http://127.0.0.1:5000
```

---

## 🖥️ Web App Usage

1. Select a **city** from the dropdown
2. Pick a **date** between `2000-02-01` and `2024-12-24`
3. Click **Predict**
4. The chart shows:
   - 🟠 **Solid line** → Actual temperature (ground truth)
   - 🟠 **Dashed line** → Predicted temperature
   - 🔵 **Solid line** → Actual Tmin
   - 🔵 **Dashed line** → Predicted Tmin

---

## 📈 Training Details

| Hyperparameter | Value |
|---|---|
| Input window | 30 days |
| Output window | 7 days |
| Hidden size | 128 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Batch size | 64 |
| Epochs | 50 |
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE |
| LR Scheduler | ReduceLROnPlateau (patience=5) |
| Gradient clipping | max_norm=1.0 |

---

## 🛠️ PyTorch Concepts Demonstrated

- Custom `nn.Module` subclass
- `torch.utils.data.Dataset` and `DataLoader`
- LSTM from scratch with stacked layers
- `model.train()` / `model.eval()` modes
- `torch.no_grad()` for inference
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
- `torch.nn.utils.clip_grad_norm_`
- `model.state_dict()` save/load
- Device-agnostic code (CPU/GPU)
- `ReduceLROnPlateau` scheduler

---

## 👥 Team

Built as part of a Deep Learning course project demonstrating PyTorch usage through real-world multivariate time series forecasting.
