# app/app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys
import os

# Add src/ to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_data, load_scalers
from evaluate import load_model, predict

app = Flask(__name__)

# These are loaded when Flask starts, kept in memory
# Loading model/data on every request would be extremely slow

BASE = os.path.join(os.path.dirname(__file__), '..')


print("Loading data...")
DF      = load_data(os.path.join(BASE, "data", "india_weather.csv"))


print("Loading scalers...")
SCALERS = load_scalers(os.path.join(BASE, "checkpoints", "scalers"))

print("Loading model...")
MODEL, DEVICE = load_model(os.path.join(BASE, "checkpoints", "lstm_weather.pt"))

CITIES = sorted(DF['city'].unique().tolist())

print(f"Ready! Cities: {CITIES}")


@app.route('/')
def index():
    # Passes city list to HTML so dropdown can be populated dynamically
    return render_template('index.html', cities=CITIES)


@app.route('/predict', methods=['POST'])
def predict_route():
    
    # Parse JSON from the frontend fetch() call
    data = request.get_json()
    city = data.get('city')
    date = data.get('date')
    
    # Basic validation
    if not city or not date:
        return jsonify({'error': 'city and date are required'}), 400
    
    if city not in CITIES:
        return jsonify({'error': f'Unknown city: {city}'}), 400
    
    try:
        # Call our evaluate function — does all the heavy lifting
        result = predict(
            city    = city,
            date    = date,
            df      = DF,
            scalers = SCALERS,
            model   = MODEL,
            device  = DEVICE
        )
        return jsonify(result)
    
    except ValueError as e:
        # Catches our date-out-of-range errors from evaluate.py
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


    