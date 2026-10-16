"""
End-to-end pipeline: Simulator → Inference API → Dashboard

Reads sensor data, sends it to the inference API for predictions,
and pushes results to the monitoring dashboard.
"""

import time
import requests
import pandas as pd

INFERENCE_URL = "http://localhost:5001/predict"
DASHBOARD_URL = "http://localhost:5002/api/ingest"

def run_pipeline():
    columns = ['engine_id', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv('data/raw/train_FD001.txt', sep=r'\s+', header=None, names=columns)

    # Simulate 3 engines at different life stages simultaneously
    configs = [
        {"engine_id": 1, "start": 170, "end": 192},
        {"engine_id": 2, "start": 1,   "end": 20},
        {"engine_id": 3, "start": 160, "end": 179},
    ]

    # Interleave readings from all engines (like real-time multi-device streaming)
    readings = []
    for cfg in configs:
        engine_data = df[(df['engine_id'] == cfg['engine_id']) &
                         (df['cycle'] >= cfg['start']) &
                         (df['cycle'] <= cfg['end'])]
        for _, row in engine_data.iterrows():
            readings.append(row)

    # Sort by cycle to interleave
    readings.sort(key=lambda r: int(r['cycle']))

    print(f"Streaming {len(readings)} sensor readings to pipeline...")

    for row in readings:
        sensor_readings = {f"sensor_{i}": float(row[f"sensor_{i}"]) for i in range(1, 22)}
        payload = {
            "engine_id": int(row['engine_id']),
            "cycle": int(row['cycle']),
            "sensor_readings": sensor_readings
        }

        # Get prediction
        try:
            pred_resp = requests.post(INFERENCE_URL, json=payload, timeout=5)
            prediction = pred_resp.json()
        except Exception as e:
            print(f"  Inference API error: {e}")
            continue

        # Send to dashboard
        dash_payload = {
            "engine_id": prediction['engine_id'],
            "cycle": prediction['cycle'],
            "predicted_rul": prediction['predicted_rul'],
            "urgency": prediction['urgency'],
            "trend": "STREAMING"
        }
        try:
            requests.post(DASHBOARD_URL, json=dash_payload, timeout=5)
        except Exception as e:
            print(f"  Dashboard error: {e}")

        print(f"  Engine {int(row['engine_id'])} | Cycle {int(row['cycle'])} | "
              f"RUL: {prediction['predicted_rul']:.1f} | {prediction['urgency']}")

        time.sleep(0.3)

    print("\nPipeline complete! View dashboard at http://localhost:5002")

if __name__ == "__main__":
    run_pipeline()