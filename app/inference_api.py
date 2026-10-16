"""
Real-time Inference API for Predictive Maintenance

This Flask app serves the trained Random Forest model as a REST API.
In production, this would be deployed as an Azure ML Managed Endpoint
or a Databricks Model Serving endpoint.

Flow:
  Sensor data (JSON) → Feature engineering → Scale → Predict RUL → Response
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime, timezone
from collections import defaultdict

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. LOAD MODEL ARTIFACTS
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_artifacts')

with open(os.path.join(ARTIFACTS_DIR, 'rf_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, 'feature_cols.json'), 'r') as f:
    feature_cols = json.load(f)

print(f"Model loaded: {type(model).__name__}")
print(f"Features expected: {len(feature_cols)}")

# ---------------------------------------------------------------------------
# 2. IN-MEMORY SENSOR HISTORY (for rolling features)
# ---------------------------------------------------------------------------
# In production, this would be Redis or Azure Cache.
# We need the last 5 readings per engine to compute rolling features.
sensor_history = defaultdict(list)

SENSOR_COLS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (mirrors Databricks pipeline)
# ---------------------------------------------------------------------------
def compute_features(engine_id: str, sensor_readings: dict) -> pd.DataFrame:
    """
    Compute the same features used during training.
    
    This MUST match the feature engineering in 02_feature_engineering exactly,
    otherwise you get training-serving skew (a common production bug).
    """
    # Extract raw sensor values
    current = {s: sensor_readings.get(s, 0.0) for s in SENSOR_COLS}

    # Store in history
    sensor_history[engine_id].append(current)
    # Keep only last 5 readings
    if len(sensor_history[engine_id]) > 5:
        sensor_history[engine_id] = sensor_history[engine_id][-5:]

    history = sensor_history[engine_id]
    history_df = pd.DataFrame(history)

    # Build feature vector
    features = {}

    # Raw sensor values
    for s in SENSOR_COLS:
        features[s] = current[s]

    # Rolling mean and std (window=5)
    for s in SENSOR_COLS:
        features[f"{s}_rolling_mean_5"] = history_df[s].mean()
        features[f"{s}_rolling_std_5"] = history_df[s].std() if len(history_df) > 1 else 0.0

    # Replace NaN with 0 (same as training)
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.fillna(0)

    # Ensure column order matches training
    feature_df = feature_df[feature_cols]

    return feature_df


# ---------------------------------------------------------------------------
# 4. API ENDPOINTS
# ---------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_type": type(model).__name__,
        "features_count": len(feature_cols),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Remaining Useful Life for incoming sensor data.
    
    Expected JSON format (matches our IoT simulator output):
    {
        "engine_id": 1,
        "cycle": 50,
        "timestamp": "2026-10-16T...",
        "sensor_readings": {
            "sensor_1": 518.67,
            "sensor_2": 642.15,
            ...
        }
    }
    """
    try:
        data = request.get_json()

        engine_id = str(data['engine_id'])
        cycle = data.get('cycle', 0)
        sensor_readings = data['sensor_readings']

        # Compute features (same pipeline as training)
        features_df = compute_features(engine_id, sensor_readings)

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predict RUL
        predicted_rul = model.predict(features_scaled)[0]
        predicted_rul = max(0, predicted_rul)  # RUL can't be negative

        # Determine urgency level
        if predicted_rul <= 15:
            urgency = "CRITICAL"
            action = "Schedule immediate maintenance"
        elif predicted_rul <= 50:
            urgency = "WARNING"
            action = "Plan maintenance within next scheduled window"
        else:
            urgency = "NORMAL"
            action = "No immediate action required"

        response = {
            "engine_id": data['engine_id'],
            "cycle": cycle,
            "predicted_rul": round(float(predicted_rul), 1),
            "urgency": urgency,
            "recommended_action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "1.0"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict RUL for multiple readings at once."""
    try:
        data = request.get_json()
        readings = data.get('readings', [])
        results = []

        for reading in readings:
            engine_id = str(reading['engine_id'])
            sensor_readings = reading['sensor_readings']
            features_df = compute_features(engine_id, sensor_readings)
            features_scaled = scaler.transform(features_df)
            predicted_rul = max(0, model.predict(features_scaled)[0])

            if predicted_rul <= 15:
                urgency = "CRITICAL"
            elif predicted_rul <= 50:
                urgency = "WARNING"
            else:
                urgency = "NORMAL"

            results.append({
                "engine_id": reading['engine_id'],
                "cycle": reading.get('cycle', 0),
                "predicted_rul": round(float(predicted_rul), 1),
                "urgency": urgency
            })

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------------
# 5. RUN SERVER
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)