import time
from flask import Flask, jsonify, render_template, request
import psutil
import mysql.connector
from mysql.connector import Error
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

app = Flask(__name__)

# ---------- Configuration ----------
MYSQL_HOST = 'localhost'
MYSQL_DB = 'metrics_db'
MYSQL_USER = 'metrics_user'
MYSQL_PASSWORD = 'your_password'   # Replace with your actual password

# Global variable to store the latest metrics
latest_metrics = {
    'cpu_util_percent': 0.0,
    'mem_util_percent': 0.0,
    'net_in': 0.0,
    'net_out': 0.0,
    'disk_io_percent': 0.0
}

# For network rate calculation
last_net_io = psutil.net_io_counters()
last_net_time = time.time()

# ---------- Database Helpers ----------
def get_db_connection():
    """Create and return a MySQL connection."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            database=MYSQL_DB,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def init_db():
    """Create the metrics table if it doesn't exist."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_util_percent FLOAT,
                mem_util_percent FLOAT,
                net_in FLOAT,
                net_out FLOAT,
                disk_io_percent FLOAT
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()

def insert_metrics(cpu, mem, net_in, net_out, disk):
    """Insert a metrics record into the database."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO metrics (cpu_util_percent, mem_util_percent, net_in, net_out, disk_io_percent)
                VALUES (%s, %s, %s, %s, %s)
            ''', (cpu, mem, net_in, net_out, disk))
            conn.commit()
            cursor.close()
        except Error as e:
            print(f"Error inserting metrics: {e}")
        finally:
            conn.close()

# ---------- Database Query for Paginated Results ----------
def get_metrics_page(page=1, per_page=10, sort_by='timestamp', sort_order='desc', search=''):
    """
    Retrieve a paginated, sorted, and filtered list of metrics from the database.
    Returns a dict with 'records', 'total', 'page', 'per_page', 'total_pages'.
    """
    conn = get_db_connection()
    if not conn:
        return {'records': [], 'total': 0, 'page': page, 'per_page': per_page, 'total_pages': 0}

    cursor = conn.cursor(dictionary=True)  # returns rows as dicts

    # Allowed sort columns to prevent SQL injection
    allowed_sort_cols = ['timestamp', 'cpu_util_percent', 'mem_util_percent',
                         'net_in', 'net_out', 'disk_io_percent']
    if sort_by not in allowed_sort_cols:
        sort_by = 'timestamp'

    sort_order = 'ASC' if sort_order.upper() == 'ASC' else 'DESC'

    # Base query with optional search filter
    base_query = "FROM metrics"
    params = []

    if search:
        # Simple global search across all columns (convert numbers to char for searching)
        base_query += """ WHERE 
            CAST(cpu_util_percent AS CHAR) LIKE %s OR
            CAST(mem_util_percent AS CHAR) LIKE %s OR
            CAST(net_in AS CHAR) LIKE %s OR
            CAST(net_out AS CHAR) LIKE %s OR
            CAST(disk_io_percent AS CHAR) LIKE %s OR
            timestamp LIKE %s"""
        search_pattern = f"%{search}%"
        params = [search_pattern] * 6

    # Count total records (for pagination)
    count_query = f"SELECT COUNT(*) as total {base_query}"
    cursor.execute(count_query, params)
    total = cursor.fetchone()['total']

    # Calculate offset
    offset = (page - 1) * per_page
    total_pages = (total + per_page - 1) // per_page

    # Main data query with sorting and pagination
    data_query = f"""
        SELECT id, timestamp, cpu_util_percent, mem_util_percent,
               net_in, net_out, disk_io_percent
        {base_query}
        ORDER BY {sort_by} {sort_order}
        LIMIT %s OFFSET %s
    """
    # Add pagination parameters
    pagination_params = params + [per_page, offset]
    cursor.execute(data_query, pagination_params)
    records = cursor.fetchall()

    cursor.close()
    conn.close()

    return {
        'records': records,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages
    }

# ---------- Metrics Collection ----------
def collect_metrics():
    """Collect current system metrics and update the global variable and database."""
    global latest_metrics, last_net_io, last_net_time

    # CPU and memory
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent

    # Disk I/O percent – using disk usage percentage of root partition as example
    disk = psutil.disk_usage('/').percent

    # Network I/O rate (bytes per second)
    net_io = psutil.net_io_counters()
    now = time.time()
    interval = now - last_net_time
    if interval > 0:
        net_in_rate = (net_io.bytes_recv - last_net_io.bytes_recv) / interval
        net_out_rate = (net_io.bytes_sent - last_net_io.bytes_sent) / interval
    else:
        net_in_rate = net_out_rate = 0.0

    # Update for next calculation
    last_net_io = net_io
    last_net_time = now

    # Update global variable
    latest_metrics = {
        'cpu_util_percent': cpu,
        'mem_util_percent': mem,
        'net_in': net_in_rate,
        'net_out': net_out_rate,
        'disk_io_percent': disk
    }

    # Insert into database
    insert_metrics(cpu, mem, net_in_rate, net_out_rate, disk)

# ---------- Scheduler ----------
scheduler = BackgroundScheduler()
scheduler.add_job(func=collect_metrics, trigger="interval", seconds=10)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

# ---------- Flask Routes ----------
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return the latest collected metrics as JSON."""
    return jsonify(latest_metrics)

@app.route('/telemetry')
def telemetry():
    """Render the telemetry HTML page."""
    return render_template('telemetry.html')

@app.route('/api/telemetry')
def api_telemetry():
    """Return paginated JSON data for the telemetry table."""
    # Get query parameters with defaults
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    sort_by = request.args.get('sort_by', 'timestamp')
    sort_order = request.args.get('sort_order', 'desc')
    search = request.args.get('search', '')

    data = get_metrics_page(page, per_page, sort_by, sort_order, search)
    return jsonify(data)

# ---------- Initialize Database ----------
init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
