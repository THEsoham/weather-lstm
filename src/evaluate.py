# src/evaluate.py

import torch
import numpy as np
import pandas as pd

from utils import load_data, load_scalers, get_window, FEATURE_COLS, TARGET_COLS
from model import WeatherLSTM

def load_model(checkpoint_path="checkpoints/lstm_weather.pt", device=None):
    """
    Instantiates the model architecture and loads saved weights.
    Must match exact hyperparameters used during training.
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate with SAME architecture as training
    model = WeatherLSTM(
        input_size   = 10,
        hidden_size  = 128,
        num_layers   = 2,
        output_steps = 7,
        output_size  = 2,
        dropout      = 0.2
    ).to(device)
    
    # Load saved weights into the model
    # map_location ensures weights load correctly regardless of 
    # whether model was trained on GPU but inference runs on CPU
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    
    # Switch to eval mode — disables dropout for deterministic predictions
    model.eval()
    
    return model, device

def inverse_transform_temps(scaled_preds, scaler):
    """
    scaled_preds : numpy array of shape (7, 2) — scaled tmax, tmin
    scaler       : the city's fitted StandardScaler (knows all 10 feature stats)
    
    Returns: numpy array of shape (7, 2) in original °C scale
    """
    
    # Find where tmax and tmin sit in the full feature list
    target_indices = [FEATURE_COLS.index(col) for col in TARGET_COLS]
    
    # Build a dummy array of zeros with all 10 features
    # shape: (7, 10)
    dummy = np.zeros((scaled_preds.shape[0], len(FEATURE_COLS)))
    
    # Plug our 2 scaled predictions into the correct columns
    dummy[:, target_indices] = scaled_preds
    
    # Inverse transform the full dummy array
    # Now the scaler can use the correct mean and std for each column
    dummy_original = scaler.inverse_transform(dummy)
    
    # Extract only the 2 temperature columns back out
    return dummy_original[:, target_indices]  # shape: (7, 2)


def predict(city, date, df, scalers, model, device):
    """
    city    : string e.g. 'Delhi'
    date    : string e.g. '2024-06-15' — the date user picks
              model predicts the 7 days AFTER this date
    df      : full loaded dataframe
    scalers : dict of per-city scalers
    model   : loaded WeatherLSTM
    device  : torch device
    
    Returns a dict with:
        'dates'      : list of 7 date strings for predicted days
        'predicted'  : { 'tmax': [...], 'tmin': [...] }
        'actual'     : { 'tmax': [...], 'tmin': [...] }
    """
    
    # ── STEP 1: fetch and scale the 30-day input window ──────────
    # shape: (30, 10) raw
    raw_window = get_window(df, city, date, window=30)
    
    # Scale using this city's scaler
    # shape: (30, 10) scaled
    scaled_window = scalers[city].transform(raw_window)
    
    # ── STEP 2: prepare tensor for model ─────────────────────────
    # Add batch dimension: (30, 10) → (1, 30, 10)
    # model expects (batch_size, seq_len, features)
    X = torch.tensor(scaled_window, dtype=torch.float32) \
             .unsqueeze(0) \
             .to(device)
    
    # ── STEP 3: run inference ─────────────────────────────────────
    with torch.no_grad():
        # shape: (1, 7, 2)
        scaled_preds = model(X)
    
    # Remove batch dimension: (1, 7, 2) → (7, 2)
    scaled_preds = scaled_preds.squeeze(0).cpu().numpy()
    
    # ── STEP 4: inverse transform back to °C ─────────────────────
    # shape: (7, 2) in original scale
    preds_celsius = inverse_transform_temps(scaled_preds, scalers[city])
    
    # ── STEP 5: fetch actual next 7 days from dataframe ──────────
    end_date   = pd.to_datetime(date)
    start_date = end_date + pd.Timedelta(days=1)
    
    city_df = df[df['city'] == city].copy()
    
    # Slice the 7 actual days after the given date
    mask = (city_df['date'] >= start_date) & \
           (city_df['date'] <= end_date + pd.Timedelta(days=7))
    
    actual_df = city_df[mask].sort_values('date').head(7)
    
    # Safety check — date too close to end of dataset
    if len(actual_df) < 7:
        raise ValueError(
            f"Not enough future data for comparison. "
            f"Pick a date before 2024-12-24."
        )
    
    # ── STEP 6: package everything for Flask ─────────────────────
    future_dates = [
        (end_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(1, 8)
    ]
    
    return {
        'city'   : city,
        'dates'  : future_dates,
        'predicted': {
            'tmax': preds_celsius[:, 0].tolist(),  # 7 values
            'tmin': preds_celsius[:, 1].tolist()   # 7 values
        },
        'actual': {
            'tmax': actual_df['temperature_2m_max'].tolist(),
            'tmin': actual_df['temperature_2m_min'].tolist()
        }
    }

