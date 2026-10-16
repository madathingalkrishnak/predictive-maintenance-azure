"""
IoT Device Simulator for Predictive Maintenance
Replays NASA C-MAPSS turbofan sensor data as real-time IoT telemetry.

Each row in the dataset becomes a JSON message sent to Azure IoT Hub,
simulating what a real sensor gateway would produce on a factory floor.
"""

import os
import json
import time
import asyncio
import pandas as pd
from datetime import datetime, timezone
from azure.iot.device.aio import IoTHubDeviceClient

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
# The connection string authenticates this "device" with your IoT Hub.
# In production, you'd use certificates or DPS (Device Provisioning Service).
# For this project, we pass it via environment variable to avoid hardcoding secrets.
CONNECTION_STRING = os.environ.get("IOTHUB_DEVICE_CONNECTION_STRING")

# How many seconds to wait between sending each sensor reading.
# Lower = faster replay. In production, sensors might send every 1-10 seconds.
SEND_INTERVAL_SECONDS = 1

# Path to the C-MAPSS training data
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'train_FD001.txt')

# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------
def load_sensor_data(file_path: str) -> pd.DataFrame:
    """
    Load the C-MAPSS dataset and return it as a DataFrame.
    
    Why this structure matters:
    - engine_id: identifies which engine (like a machine ID on a factory floor)
    - cycle: the time step (like a timestamp in real life)
    - op_settings: operating conditions the engine is running under
    - sensors: the actual readings we'll use for prediction
    """
    columns = (
        ['engine_id', 'cycle']
        + [f'op_setting_{i}' for i in range(1, 4)]
        + [f'sensor_{i}' for i in range(1, 22)]
    )
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)
    print(f"Loaded {len(df)} rows for {df['engine_id'].nunique()} engines")
    return df


# ---------------------------------------------------------------------------
# 3. MESSAGE CONSTRUCTION
# ---------------------------------------------------------------------------
def build_telemetry_message(row: pd.Series) -> dict:
    """
    Convert one row of sensor data into an IoT telemetry message.
    
    This is the JSON structure that your streaming pipeline will consume.
    In a real system, the sensor gateway would produce something very similar.
    
    We add a timestamp because the original dataset doesn't have one —
    real IoT data always has timestamps for time-series analysis.
    """
    message = {
        "engine_id": int(row["engine_id"]),
        "cycle": int(row["cycle"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operational_settings": {
            "op_setting_1": float(row["op_setting_1"]),
            "op_setting_2": float(row["op_setting_2"]),
            "op_setting_3": float(row["op_setting_3"]),
        },
        "sensor_readings": {
            f"sensor_{i}": float(row[f"sensor_{i}"])
            for i in range(1, 22)
        },
    }
    return message


# ---------------------------------------------------------------------------
# 4. SENDING TELEMETRY TO IOT HUB
# ---------------------------------------------------------------------------
async def send_telemetry(client: IoTHubDeviceClient, message: dict) -> None:
    """
    Send a single telemetry message to Azure IoT Hub.
    
    Why async? In production, you don't want to block the sensor reading
    loop while waiting for network I/O. The async SDK handles this efficiently.
    
    The message is sent as a JSON string. IoT Hub stores it and makes it
    available to downstream consumers (Stream Analytics, Event Hubs, etc.)
    """
    payload = json.dumps(message)
    await client.send_message(payload)


# ---------------------------------------------------------------------------
# 5. MAIN SIMULATION LOOP
# ---------------------------------------------------------------------------
async def simulate_engine(engine_id: int = 1, max_cycles: int = None) -> None:
    """
    Simulate one engine's lifecycle by replaying its sensor data.
    
    Args:
        engine_id: Which engine from the dataset to simulate (1-100)
        max_cycles: Optional limit on how many cycles to send (useful for testing)
    """
    if not CONNECTION_STRING:
        raise ValueError(
            "Set the IOTHUB_DEVICE_CONNECTION_STRING environment variable.\n"
            "Get it with: az iot hub device-identity connection-string show "
            "--hub-name <hub> --device-id <device> --output tsv"
        )

    # Load data and filter to the chosen engine
    df = load_sensor_data(DATA_PATH)
    engine_data = df[df['engine_id'] == engine_id].copy()
    total_cycles = len(engine_data)
    print(f"Simulating engine {engine_id}: {total_cycles} cycles until failure")

    if max_cycles:
        engine_data = engine_data.head(max_cycles)

    # Connect to IoT Hub
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    await client.connect()
    print("Connected to IoT Hub")

    try:
        for idx, (_, row) in enumerate(engine_data.iterrows()):
            message = build_telemetry_message(row)
            await send_telemetry(client, message)

            # Calculate RUL for display (we know this, but the model won't!)
            rul = total_cycles - int(row['cycle'])
            print(
                f"[Cycle {int(row['cycle'])}/{total_cycles}] "
                f"RUL={rul} | "
                f"sensor_2={message['sensor_readings']['sensor_2']:.2f}, "
                f"sensor_7={message['sensor_readings']['sensor_7']:.2f}"
            )

            await asyncio.sleep(SEND_INTERVAL_SECONDS)
    finally:
        await client.disconnect()
        print("Disconnected from IoT Hub")


# ---------------------------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IoT Device Simulator")
    parser.add_argument("--engine-id", type=int, default=1, help="Engine ID to simulate (1-100)")
    parser.add_argument("--max-cycles", type=int, default=10, help="Max cycles to send (for testing)")
    args = parser.parse_args()

    asyncio.run(simulate_engine(engine_id=args.engine_id, max_cycles=args.max_cycles))