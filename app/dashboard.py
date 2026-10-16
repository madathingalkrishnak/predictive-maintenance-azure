"""
Monitoring Dashboard for Predictive Maintenance

Displays real-time engine health, predictions, and maintenance status.
In production, this would be Azure Monitor + Grafana or a Power BI dashboard.
"""

from flask import Flask, jsonify, render_template_string
from datetime import datetime, timezone
import requests

app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Predictive Maintenance Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1923; color: #e0e0e0; }
        .header { background: #1a2332; padding: 20px 40px; border-bottom: 2px solid #2d3748; }
        .header h1 { font-size: 24px; color: #63b3ed; }
        .header p { color: #718096; margin-top: 4px; }
        .container { max-width: 1200px; margin: 0 auto; padding: 30px 40px; }
        .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #1a2332; border-radius: 10px; padding: 24px; border: 1px solid #2d3748; }
        .stat-card .label { color: #718096; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
        .stat-card .value { font-size: 36px; font-weight: 700; margin-top: 8px; }
        .stat-card.critical .value { color: #fc8181; }
        .stat-card.warning .value { color: #f6e05e; }
        .stat-card.normal .value { color: #68d391; }
        .stat-card.total .value { color: #63b3ed; }
        .section-title { font-size: 18px; color: #a0aec0; margin-bottom: 15px; border-bottom: 1px solid #2d3748; padding-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; background: #1a2332; border-radius: 10px; overflow: hidden; }
        th { background: #2d3748; padding: 12px 16px; text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #a0aec0; }
        td { padding: 12px 16px; border-bottom: 1px solid #2d3748; }
        .badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-block; }
        .badge-critical { background: rgba(252, 129, 129, 0.2); color: #fc8181; }
        .badge-warning { background: rgba(246, 224, 94, 0.2); color: #f6e05e; }
        .badge-normal { background: rgba(104, 211, 145, 0.2); color: #68d391; }
        .rul-bar { height: 8px; border-radius: 4px; background: #2d3748; overflow: hidden; width: 120px; display: inline-block; vertical-align: middle; margin-left: 10px; }
        .rul-fill { height: 100%; border-radius: 4px; }
        .btn { background: #3182ce; color: white; border: none; padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #2b6cb0; }
        .refresh-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .last-update { color: #718096; font-size: 13px; }
        .wo-section { margin-top: 30px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Predictive Maintenance Dashboard</h1>
        <p>Real-time engine health monitoring and maintenance scheduling</p>
    </div>
    <div class="container">
        <div class="refresh-bar">
            <span class="last-update" id="lastUpdate">Loading...</span>
            <button class="btn" onclick="loadData()">Refresh</button>
        </div>

        <div class="stats-row">
            <div class="stat-card total"><div class="label">Engines Tracked</div><div class="value" id="totalEngines">-</div></div>
            <div class="stat-card critical"><div class="label">Critical</div><div class="value" id="criticalCount">-</div></div>
            <div class="stat-card warning"><div class="label">Warning</div><div class="value" id="warningCount">-</div></div>
            <div class="stat-card normal"><div class="label">Normal</div><div class="value" id="normalCount">-</div></div>
        </div>

        <h2 class="section-title">Engine Status</h2>
        <table>
            <thead>
                <tr><th>Engine ID</th><th>Last Cycle</th><th>Predicted RUL</th><th>Health</th><th>Urgency</th><th>Trend</th></tr>
            </thead>
            <tbody id="engineTable"></tbody>
        </table>

        <div class="wo-section">
            <h2 class="section-title">Recent Work Orders</h2>
            <table>
                <thead>
                    <tr><th>WO ID</th><th>Engine</th><th>Priority</th><th>RUL at Creation</th><th>Window</th><th>Status</th></tr>
                </thead>
                <tbody id="woTable"></tbody>
            </table>
        </div>
    </div>

    <script>
        async function loadData() {
            try {
                const resp = await fetch('/api/dashboard');
                const data = await resp.json();

                document.getElementById('totalEngines').textContent = data.total_engines_tracked;
                document.getElementById('criticalCount').textContent = data.engines_by_urgency.CRITICAL;
                document.getElementById('warningCount').textContent = data.engines_by_urgency.WARNING;
                document.getElementById('normalCount').textContent = data.engines_by_urgency.NORMAL;
                document.getElementById('lastUpdate').textContent = 'Last updated: ' + new Date().toLocaleTimeString();

                const engineTable = document.getElementById('engineTable');
                engineTable.innerHTML = '';
                data.engines.forEach(e => {
                    const pct = Math.min(100, (e.predicted_rul / 125) * 100);
                    const color = e.urgency === 'CRITICAL' ? '#fc8181' : e.urgency === 'WARNING' ? '#f6e05e' : '#68d391';
                    const badgeClass = 'badge-' + e.urgency.toLowerCase();
                    engineTable.innerHTML += '<tr>' +
                        '<td><strong>Engine ' + e.engine_id + '</strong></td>' +
                        '<td>' + e.last_cycle + '</td>' +
                        '<td>' + e.predicted_rul.toFixed(1) +
                        '  <div class="rul-bar"><div class="rul-fill" style="width:' + pct + '%;background:' + color + '"></div></div></td>' +
                        '<td><div style="width:12px;height:12px;border-radius:50%;background:' + color + ';display:inline-block"></div></td>' +
                        '<td><span class="badge ' + badgeClass + '">' + e.urgency + '</span></td>' +
                        '<td>' + e.trend + '</td></tr>';
                });

                const woTable = document.getElementById('woTable');
                woTable.innerHTML = '';
                data.recent_work_orders.forEach(wo => {
                    const prioClass = wo.priority === 'EMERGENCY' ? 'badge-critical' : 'badge-warning';
                    woTable.innerHTML += '<tr>' +
                        '<td>' + wo.work_order_id + '</td>' +
                        '<td>Engine ' + wo.engine_id + '</td>' +
                        '<td><span class="badge ' + prioClass + '">' + wo.priority + '</span></td>' +
                        '<td>' + wo.predicted_rul + '</td>' +
                        '<td>' + wo.assigned_window + '</td>' +
                        '<td>' + wo.status + '</td></tr>';
                });
            } catch (err) {
                document.getElementById('lastUpdate').textContent = 'Error loading data: ' + err.message;
            }
        }
        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>
"""

# In-memory state (in production, this would be a shared database)
dashboard_state = {
    "engines": {},
    "work_orders": [],
    "wo_counter": 0
}


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/ingest', methods=['POST'])
def ingest_prediction():
    """Receive a prediction and update dashboard state."""
    from flask import request
    data = request.get_json()

    engine_id = str(data['engine_id'])
    dashboard_state["engines"][engine_id] = {
        "engine_id": data['engine_id'],
        "last_cycle": data.get('cycle', 0),
        "predicted_rul": data['predicted_rul'],
        "urgency": data['urgency'],
        "trend": data.get('trend', 'UNKNOWN'),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Create work order if needed
    if data['urgency'] in ('CRITICAL', 'WARNING'):
        existing = any(
            wo['engine_id'] == data['engine_id'] and wo['status'] == 'OPEN'
            for wo in dashboard_state['work_orders']
        )
        if not existing:
            dashboard_state['wo_counter'] += 1
            dashboard_state['work_orders'].append({
                "work_order_id": f"WO-{dashboard_state['wo_counter']:04d}",
                "engine_id": data['engine_id'],
                "priority": "EMERGENCY" if data['urgency'] == 'CRITICAL' else "PLANNED",
                "predicted_rul": data['predicted_rul'],
                "assigned_window": "IMMEDIATE" if data['urgency'] == 'CRITICAL' else "Next scheduled window",
                "status": "OPEN"
            })

    return jsonify({"status": "ok"})


@app.route('/api/dashboard')
def dashboard_api():
    """Return current dashboard state."""
    engines = list(dashboard_state["engines"].values())
    urgency_counts = {"CRITICAL": 0, "WARNING": 0, "NORMAL": 0}
    for e in engines:
        urgency_counts[e["urgency"]] = urgency_counts.get(e["urgency"], 0) + 1

    return jsonify({
        "total_engines_tracked": len(engines),
        "engines_by_urgency": urgency_counts,
        "engines": sorted(engines, key=lambda x: x['predicted_rul']),
        "recent_work_orders": dashboard_state["work_orders"][-10:],
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


if __name__ == '__main__':
    print("Dashboard running at http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=True)