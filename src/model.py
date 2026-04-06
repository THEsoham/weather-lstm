# src/model.py

import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, 
                 output_steps=7, output_size=2, dropout=0.2):
        """
        input_size  : number of features per timestep (10)
        hidden_size : number of memory cells in LSTM (128 — hyperparameter)
        num_layers  : stacked LSTM layers (2)
        output_steps: how many days to predict (7)
        output_size : how many targets per day (2 — tmax, tmin)
        dropout     : regularization between LSTM layers
        """
        super(WeatherLSTM, self).__init__()
        
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_steps = output_steps
        self.output_size  = output_size


        # --- LSTM LAYER ---
        # batch_first=True means input shape is (batch, seq_len, features)
        # instead of PyTorch's default (seq_len, batch, features)
        # Much more intuitive to work with
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout    # applied between stacked layers (not after last)
        )
        
        # --- FULLY CONNECTED HEAD ---
        # After LSTM processes 30 days, we take the LAST hidden state h_30
        # shape: (batch, hidden_size) = (batch, 128)
        # We need to project this to (output_steps * output_size) = 7 * 2 = 14 values
        # then reshape to (batch, 7, 2)
        self.fc = nn.Linear(hidden_size, output_steps * output_size)


    def forward(self, x):
        """
        x shape: (batch_size, 30, 10)
                  batch of 30-day windows, 10 features each
        """
        
        # --- INITIALIZE HIDDEN AND CELL STATES TO ZERO ---
        # h0, c0 shape: (num_layers, batch_size, hidden_size)
        # We create them here so they move to whatever device x is on
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # --- LSTM FORWARD PASS ---
        # lstm_out shape: (batch_size, 30, hidden_size)
        #   — hidden state at every timestep, useful for attention later
        # (hn, cn) — final hidden and cell states after day 30
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # --- TAKE ONLY THE LAST TIMESTEP'S OUTPUT ---
        # We only care about h_30 — the summary after seeing all 30 days
        # shape: (batch_size, hidden_size) = (batch_size, 128)
        last_hidden = lstm_out[:, -1, :]
        
        # --- PROJECT TO OUTPUT ---
        # shape: (batch_size, 14)
        out = self.fc(last_hidden)
        
        # --- RESHAPE TO (batch, 7, 2) ---
        # 7 days × 2 targets — clean interpretable output
        out = out.view(batch_size, self.output_steps, self.output_size)
        
        return out  # shape: (batch_size, 7, 2)
    
    