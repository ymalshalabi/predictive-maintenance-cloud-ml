import time
from flask import Flask, jsonify, render_template, request
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

app = Flask(__name__)

# ---------- Configuration ----------
NODE_IPS = [
    '13.50.208.58',
    '13.62.95.174',
    '13.62.215.163'
]
NODE_PORT = 5000
HEALTH_ENDPOINT = '/metrics'
REQUEST_TIMEOUT = 3

node_status = {ip: 'critical' for ip in NODE_IPS}

# ---------- Health Check ----------
def check_nodes():
    global node_status
    for ip in NODE_IPS:
        url = f'http://{ip}:{NODE_PORT}{HEALTH_ENDPOINT}'
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                node_status[ip] = 'active'
            else:
                node_status[ip] = 'critical'
        except requests.exceptions.RequestException:
            node_status[ip] = 'critical'
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Node status updated")

scheduler = BackgroundScheduler()
scheduler.add_job(func=check_nodes, trigger="interval", seconds=30)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())
check_nodes()  # initial run

# ---------- Routes ----------
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/nodes')
def api_nodes():
    nodes = [{'ip': ip, 'status': node_status.get(ip, 'critical')} for ip in NODE_IPS]
    return jsonify(nodes)

# ---------- NEW: Proxy endpoint for node metrics ----------
@app.route('/api/node-metrics/<ip>')
def proxy_node_metrics(ip):
    """
    Fetch metrics from the specified node's telemetry API and return them.
    """
    # Optional: validate that ip is in NODE_IPS for security
    if ip not in NODE_IPS:
        return jsonify({'error': 'Invalid node IP'}), 400

    try:
        # Use the same parameters as the frontend would (last 20 records)
        url = f'http://{ip}:{NODE_PORT}/api/telemetry?per_page=20&sort_by=timestamp&sort_order=desc'
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return jsonify(resp.json())
        else:
            return jsonify({'error': f'Node returned status {resp.status_code}'}), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
