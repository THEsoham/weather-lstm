# src/utils.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# The features we'll feed into the LSTM as input
FEATURE_COLS = [
    'temperature_2m_max', 'temperature_2m_min',
    'apparent_temperature_max', 'apparent_temperature_min',
    'precipitation_sum', 'rain_sum', 'weather_code',
    'wind_speed_10m_max', 'wind_gusts_10m_max',
    'wind_direction_10m_dominant'
]

# The two columns we want to PREDICT (subset of features above)
TARGET_COLS = ['temperature_2m_max', 'temperature_2m_min']


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Parse date column so we can sort and slice by date later
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by city then date — critical for correct 30-day windows
    df = df.sort_values(['city', 'date']).reset_index(drop=True)
    
    return df


def fit_scalers(df: pd.DataFrame, save_dir: str = "checkpoints/scalers") -> dict:
    """
    Fits one StandardScaler per city on all its historical data.
    Returns a dict like: { 'Delhi': <fitted_scaler>, 'Mumbai': <fitted_scaler>, ... }
    """
    
    # Create the directory to save scalers if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    scalers = {}
    
    for city, group in df.groupby('city'):
        # Extract only the feature columns for this city
        city_features = group[FEATURE_COLS].values  # shape: (N_days, 10)
        
        # Fit scaler on ALL historical data for this city
        scaler = StandardScaler()
        scaler.fit(city_features)
        
        scalers[city] = scaler
        
        # Save each scaler to disk — Flask will load these at inference time
        scaler_path = os.path.join(save_dir, f"{city}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    return scalers


def load_scalers(save_dir: str = "checkpoints/scalers") -> dict:
    """
    Loads all pre-fitted city scalers from disk.
    Called once when Flask app starts up.
    """
    
    scalers = {}
    
    for fname in os.listdir(save_dir):
        if fname.endswith('_scaler.pkl'):
            # Extract city name from filename e.g. 'Delhi_scaler.pkl' -> 'Delhi'
            city = fname.replace('_scaler.pkl', '')
            with open(os.path.join(save_dir, fname), 'rb') as f:
                scalers[city] = pickle.load(f)
    
    return scalers


def get_window(df: pd.DataFrame, city: str, end_date: str,
               window: int = 30) -> np.ndarray:
    """
    Fetches the last `window` days of data for a city ending at end_date.
    Returns raw (unscaled) feature array of shape (window, 10)
    """
    
    end_date = pd.to_datetime(end_date)
    
    # Filter to this city only
    city_df = df[df['city'] == city].copy()
    
    # Get rows where date falls in (end_date - 30 days, end_date]
    mask = (city_df['date'] <= end_date) & \
           (city_df['date'] > end_date - pd.Timedelta(days=window))
    
    window_df = city_df[mask].sort_values('date')
    
    # Safety check — if user picks a date too early in 2000, window will be short
    if len(window_df) < window:
        raise ValueError(
            f"Not enough data: got {len(window_df)} days, need {window}. "
            f"Pick a date after {pd.Timestamp('2000-02-01').date()}"
        )
    
    return window_df[FEATURE_COLS].values  # shape: (30, 10)