"""
Maintenance Scheduling Application

Consumes predictions from the inference API and optimizes maintenance tasks.
In a real factory, this would integrate with a CMMS (Computerized Maintenance
Management System) like SAP PM or Maximo.

Key scheduling logic:
- CRITICAL (RUL ≤ 15): Immediate work order, stop equipment if possible
- WARNING (RUL 16-50): Schedule in next maintenance window
- NORMAL (RUL > 50): Monitor, no action needed

It also tracks prediction history per engine to detect degradation trends.
"""

import json
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
INFERENCE_API_URL = "http://localhost:5001/predict"

# Maintenance windows — in production, these come from the factory calendar
MAINTENANCE_WINDOWS = [
    {"day": "Monday", "shift": "Night", "capacity": 3},    # Can service 3 engines
    {"day": "Wednesday", "shift": "Night", "capacity": 3},
    {"day": "Saturday", "shift": "Day", "capacity": 5},
]


# ---------------------------------------------------------------------------
# 2. MAINTENANCE TRACKER
# ---------------------------------------------------------------------------
class MaintenanceScheduler:
    """
    Tracks engine health and schedules maintenance tasks.
    
    In production, this state would be in a database (e.g., Azure SQL or Cosmos DB).
    For this project, we keep it in memory to focus on the logic.
    """

    def __init__(self):
        # Prediction history per engine
        self.engine_history = defaultdict(list)
        # Active maintenance work orders
        self.work_orders = []
        # Completed maintenance
        self.maintenance_log = []
        # Work order counter
        self.wo_counter = 0

    def process_prediction(self, prediction: dict) -> dict:
        """
        Process a prediction and decide what action to take.
        
        Returns a maintenance decision with details.
        """
        engine_id = prediction['engine_id']
        rul = prediction['predicted_rul']
        urgency = prediction['urgency']
        cycle = prediction.get('cycle', 0)

        # Store in history
        self.engine_history[engine_id].append({
            "cycle": cycle,
            "rul": rul,
            "urgency": urgency,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Check degradation trend (RUL decreasing over last 5 readings)
        trend = self._compute_trend(engine_id)

        decision = {
            "engine_id": engine_id,
            "current_rul": rul,
            "urgency": urgency,
            "trend": trend,
            "work_order": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Decision logic
        if urgency == "CRITICAL":
            wo = self._create_work_order(engine_id, rul, "EMERGENCY",
                "Immediate inspection and repair required. "
                "Equipment at high risk of failure."
            )
            decision["work_order"] = wo
            decision["action"] = "EMERGENCY_WORK_ORDER_CREATED"

        elif urgency == "WARNING":
            # Check if a work order already exists for this engine
            existing = self._get_active_work_order(engine_id)
            if existing:
                decision["work_order"] = existing
                decision["action"] = "EXISTING_WORK_ORDER_ACTIVE"
            else:
                next_window = self._find_next_maintenance_window()
                wo = self._create_work_order(engine_id, rul, "PLANNED",
                    f"Schedule maintenance in next window: {next_window}. "
                    f"Degradation detected with RUL={rul:.0f} cycles remaining."
                )
                decision["work_order"] = wo
                decision["action"] = "PLANNED_WORK_ORDER_CREATED"

        elif trend == "DEGRADING" and rul < 80:
            decision["action"] = "MONITORING_INCREASED"
            decision["note"] = (
                f"Engine showing degradation trend. "
                f"RUL={rul:.0f}. Increasing monitoring frequency."
            )
        else:
            decision["action"] = "NO_ACTION"

        return decision

    def _compute_trend(self, engine_id) -> str:
        """Analyze if RUL is trending downward."""
        history = self.engine_history[engine_id]
        if len(history) < 3:
            return "INSUFFICIENT_DATA"

        recent = [h['rul'] for h in history[-5:]]

        # Check if RUL is consistently decreasing
        decreasing_count = sum(
            1 for i in range(1, len(recent))
            if recent[i] < recent[i-1]
        )

        if decreasing_count >= len(recent) - 1:
            return "DEGRADING"
        elif recent[-1] > recent[0]:
            return "STABLE"
        else:
            return "FLUCTUATING"

    def _create_work_order(self, engine_id, rul, priority, description) -> dict:
        """Create a maintenance work order."""
        self.wo_counter += 1
        wo = {
            "work_order_id": f"WO-{self.wo_counter:04d}",
            "engine_id": engine_id,
            "priority": priority,
            "description": description,
            "predicted_rul": round(rul, 1),
            "status": "OPEN",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "assigned_window": self._find_next_maintenance_window() if priority == "PLANNED" else "IMMEDIATE"
        }
        self.work_orders.append(wo)
        return wo

    def _get_active_work_order(self, engine_id) -> dict:
        """Check if there's already an open work order for this engine."""
        for wo in self.work_orders:
            if wo['engine_id'] == engine_id and wo['status'] == 'OPEN':
                return wo
        return None

    def _find_next_maintenance_window(self) -> str:
        """Find the next available maintenance window."""
        # Simple rotation — in production, check actual calendar and capacity
        today = datetime.now(timezone.utc)
        for i in range(7):
            check_date = today + timedelta(days=i)
            day_name = check_date.strftime("%A")
            for window in MAINTENANCE_WINDOWS:
                if window['day'] == day_name:
                    return f"{day_name} {window['shift']} shift ({check_date.strftime('%Y-%m-%d')})"
        return "Next available window"

    def get_dashboard_summary(self) -> dict:
        """Get a summary of all engine statuses — for monitoring dashboards."""
        summary = {
            "total_engines_tracked": len(self.engine_history),
            "open_work_orders": len([wo for wo in self.work_orders if wo['status'] == 'OPEN']),
            "engines_by_urgency": {"CRITICAL": 0, "WARNING": 0, "NORMAL": 0},
            "recent_work_orders": self.work_orders[-5:],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Get latest urgency per engine
        for engine_id, history in self.engine_history.items():
            if history:
                latest_urgency = history[-1]['urgency']
                summary["engines_by_urgency"][latest_urgency] += 1

        return summary


# ---------------------------------------------------------------------------
# 3. DEMO: End-to-end simulation
# ---------------------------------------------------------------------------
def run_demo():
    """
    Demonstrate the full pipeline with consecutive cycles
    so rolling features build up properly (like real streaming).
    """
    import pandas as pd

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE SCHEDULER — DEMO")
    print("=" * 60)

    scheduler = MaintenanceScheduler()

    columns = ['engine_id', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv('data/raw/train_FD001.txt', sep=r'\s+', header=None, names=columns)

    # Simulate engines by sending CONSECUTIVE cycles
    # This mimics real streaming where every reading arrives in order
    demo_configs = [
        {"engine_id": 1, "start_cycle": 170, "end_cycle": 192},  # Last 22 cycles → approaching failure
        {"engine_id": 2, "start_cycle": 1,   "end_cycle": 20},   # First 20 cycles → healthy
        {"engine_id": 3, "start_cycle": 160, "end_cycle": 179},  # Last 19 cycles → approaching failure
    ]

    for config in demo_configs:
        engine_id = config['engine_id']
        engine_data = df[df['engine_id'] == engine_id]
        max_cycle = engine_data['cycle'].max()

        cycle_range = engine_data[
            (engine_data['cycle'] >= config['start_cycle']) &
            (engine_data['cycle'] <= config['end_cycle'])
        ]

        print(f"\n{'─'*60}")
        print(f"ENGINE {engine_id} (total lifecycle: {max_cycle} cycles)")
        print(f"Streaming cycles {config['start_cycle']} to {config['end_cycle']}")
        print(f"{'─'*60}")

        for _, row in cycle_range.iterrows():
            cycle = int(row['cycle'])
            sensor_readings = {
                f"sensor_{i}": float(row[f"sensor_{i}"])
                for i in range(1, 22)
            }

            payload = {
                "engine_id": engine_id,
                "cycle": cycle,
                "sensor_readings": sensor_readings
            }

            try:
                response = requests.post(INFERENCE_API_URL, json=payload, timeout=5)
                prediction = response.json()
            except requests.exceptions.ConnectionError:
                print(f"  ERROR: Cannot connect to inference API at {INFERENCE_API_URL}")
                print(f"  Make sure the API is running: python3 app/inference_api.py")
                return

            decision = scheduler.process_prediction(prediction)

            actual_rul = max_cycle - cycle

            # Print every 3rd cycle to keep output readable, plus first and last
            is_notable = (
                cycle == config['start_cycle'] or
                cycle == config['end_cycle'] or
                cycle % 3 == 0 or
                prediction['urgency'] != 'NORMAL'
            )

            if is_notable:
                print(f"\n  Cycle {cycle:>3d} | Actual RUL: {actual_rul:>3d} | "
                      f"Predicted RUL: {prediction['predicted_rul']:>5.1f} | "
                      f"Urgency: {prediction['urgency']}")
                print(f"           | Trend: {decision['trend']} | "
                      f"Action: {decision['action']}")

                if decision.get('work_order'):
                    wo = decision['work_order']
                    print(f"           | Work Order: {wo['work_order_id']} "
                          f"({wo['priority']}) - {wo['assigned_window']}")

            time.sleep(0.2)

    # Final dashboard
    summary = scheduler.get_dashboard_summary()
    print(f"\n{'='*60}")
    print("DASHBOARD SUMMARY")
    print(f"{'='*60}")
    print(f"  Engines tracked:    {summary['total_engines_tracked']}")
    print(f"  Open work orders:   {summary['open_work_orders']}")
    print(f"  Critical engines:   {summary['engines_by_urgency']['CRITICAL']}")
    print(f"  Warning engines:    {summary['engines_by_urgency']['WARNING']}")
    print(f"  Normal engines:     {summary['engines_by_urgency']['NORMAL']}")

    if summary['recent_work_orders']:
        print(f"\n  Recent Work Orders:")
        for wo in summary['recent_work_orders']:
            print(f"    {wo['work_order_id']}: Engine {wo['engine_id']} "
                  f"| {wo['priority']} | RUL={wo['predicted_rul']} "
                  f"| {wo['status']}")


if __name__ == "__main__":
    run_demo()