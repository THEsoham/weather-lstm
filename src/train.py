# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from utils import load_data, fit_scalers, FEATURE_COLS, TARGET_COLS
from dataset import WeatherDataset
from model import WeatherLSTM


# --- HYPERPARAMETERS ---
# Keeping them all here makes experimentation easy

CSV_PATH      = r"P:\weather-lstm-project\data\india_weather.csv"  # path to raw data
CHECKPOINT    = r"P:\weather-lstm-project\checkpoints\lstm_weather.pt"  # where to save the best model
INPUT_WINDOW  = 30       # days of history fed to model
OUTPUT_WINDOW = 7        # days to predict
INPUT_SIZE    = 10       # number of features
HIDDEN_SIZE   = 128      # LSTM memory cells
NUM_LAYERS    = 2        # stacked LSTM layers
DROPOUT       = 0.2      # regularization
BATCH_SIZE    = 64       # samples per gradient update
EPOCHS        = 50       # full passes through dataset
LR            = 1e-3     # Adam learning rate
VAL_SPLIT     = 0.1      # 10% of data for validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

def train():
    
    # --- DATA LOADING ---
    print("Loading data...")
    df = load_data(CSV_PATH)
    
    # Fit per-city scalers and save them to disk
    scalers = fit_scalers(df)
    
    # Build the full dataset with sliding windows
    dataset = WeatherDataset(df, scalers, INPUT_WINDOW, OUTPUT_WINDOW)
    print(f"Total samples: {len(dataset)}")
    
    # --- TRAIN / VAL SPLIT ---
    # random_split randomly assigns samples to train and val
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # DataLoader handles batching, shuffling, and parallel loading
    # shuffle=True so model doesn't memorize the order of cities/dates
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


    # --- MODEL SETUP ---
    model = WeatherLSTM(
        input_size   = INPUT_SIZE,
        hidden_size  = HIDDEN_SIZE,
        num_layers   = NUM_LAYERS,
        output_steps = OUTPUT_WINDOW,
        output_size  = 2,           # tmax, tmin
        dropout      = DROPOUT
    ).to(device)  # move all model parameters to GPU/CPU
    
    # MSE loss — penalizes large temperature errors heavily
    criterion = nn.MSELoss()
    
    # Adam optimizer — pass model parameters so it knows what to update
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Learning rate scheduler — halves LR if val loss plateaus for 5 epochs
    # Helps squeeze out last bits of performance in later epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
    
    # Track best validation loss to save the best model checkpoint
    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)


    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        
        # ── TRAIN PHASE ──────────────────────────────────────────
        model.train()  # enables dropout, batch norm (if any)
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            
            # Move batch to same device as model
            X_batch = X_batch.to(device)  # (64, 30, 10)
            y_batch = y_batch.to(device)  # (64,  7,  2)
            
            # CRITICAL: clear gradients from previous batch
            # PyTorch ACCUMULATES gradients by default — must reset each step
            optimizer.zero_grad()
            
            # 1. Forward pass
            predictions = model(X_batch)           # (64, 7, 2)
            
            # 2. Compute loss
            loss = criterion(predictions, y_batch)
            
            # 3. Backward pass — computes d(loss)/d(every parameter)
            loss.backward()
            
            # Gradient clipping — prevents exploding gradients in LSTM
            # If gradient norm exceeds 1.0, scale it down
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 4. Update weights
            optimizer.step()
            
            train_loss += loss.item()  # .item() extracts scalar from tensor
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ── VALIDATION PHASE ─────────────────────────────────────
        model.eval()  # disables dropout — deterministic predictions
        val_loss = 0.0
        
        with torch.no_grad():  # no gradient computation — saves memory and speed
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler with current val loss
        scheduler.step(avg_val_loss)
        
        # ── CHECKPOINT — save if best val loss so far ─────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT}")


# Entry point
if __name__ == "__main__":
    train()


    