# src/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from utils import FEATURE_COLS, TARGET_COLS


class WeatherDataset(Dataset):
    def __init__(self, df, scalers, input_window=30, output_window=7):
        """
        df           : full dataframe (all cities)
        scalers      : dict of per-city fitted StandardScalers
        input_window : how many past days we feed in (30)
        output_window: how many future days we predict (7)
        """
        
        self.input_window = input_window
        self.output_window = output_window
        
        # We'll store all samples here as a list of (X, y) tuples
        # X shape: (30, 10)   — 30 days, 10 features
        # y shape: (7,  2)    — 7 days, 2 temp targets
        self.samples = []
        
        self._build_samples(df, scalers)

    def _build_samples(self, df, scalers):
        
        # Process each city independently
        for city, group in df.groupby('city'):
            
            # Sort by date just to be safe
            group = group.sort_values('date').reset_index(drop=True)
            
            # Scale ALL features using this city's fitted scaler
            # shape: (N_days, 10)
            scaled = scalers[city].transform(group[FEATURE_COLS].values)
            
            # Find indices of TARGET_COLS inside FEATURE_COLS
            # e.g. temperature_2m_max is index 0, temperature_2m_min is index 1
            target_indices = [FEATURE_COLS.index(col) for col in TARGET_COLS]
            
            total_days = len(scaled)
            
            # Slide the window across the entire city timeline
            for start in range(total_days - self.input_window - self.output_window + 1):
                
                end_input  = start + self.input_window       # 30 days
                end_target = end_input + self.output_window  # next 7 days
                
                # Input: full 30 days, all 10 features
                X = scaled[start : end_input]                # shape: (30, 10)
                
                # Target: next 7 days, only the 2 temp columns
                y = scaled[end_input : end_target, target_indices]  # shape: (7, 2)
                
                self.samples.append((
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32)
                ))

    def __len__(self):
        # PyTorch calls this to know how many samples exist
        return len(self.samples)
    
    def __getitem__(self, idx):
        # PyTorch calls this to fetch the i-th sample
        # Returns: X of shape (30, 10), y of shape (7, 2)
        return self.samples[idx]