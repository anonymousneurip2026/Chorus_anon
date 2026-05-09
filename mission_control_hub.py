import os
import glob
import time
from flask import Flask, send_file, jsonify

app = Flask(__name__)

# Configuration
LOG_DIR = 'logs'
RESULTS_DIR = 'results'
DASHBOARD_FILE = 'mission_control.html'

def get_latest_log():
    log_files = glob.glob(os.path.join(LOG_DIR, '*.log'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)

@app.route('/')
def index():
    return send_file(DASHBOARD_FILE)

@app.route('/api/logs')
def get_logs():
    latest_log = get_latest_log()
    if not latest_log:
        return jsonify({"lines": ["No logs found. Start a sweep to see activity."]})
    
    try:
        with open(latest_log, 'r') as f:
            # Get last 50 lines
            lines = f.readlines()[-50:]
            return jsonify({"lines": [l.strip() for l in lines], "filename": os.path.basename(latest_log)})
    except Exception as e:
        return jsonify({"lines": [f"Error reading log: {str(e)}"]})

@app.route('/api/stats')
def get_stats():
    # Simulated stats parsing logic
    # In production, this would crawl the results/ directory for metrics.csv
    return jsonify({
        "lung_auc": 0.965,
        "rcc_auc": 0.951,
        "progress": 75,
        "active_variant": "v4_hybrid"
    })

if __name__ == '__main__':
    print(f"--- CHORUS MISSION CONTROL HUB ---")
    print(f"Serving at: http://localhost:5000")
    print(f"Logs directory: {LOG_DIR}")
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    app.run(host='0.0.0.0', port=5000, debug=False)
