# Copyright (c) 2026 Yasmin Mazen AlShalabi
# SPDX-License-Identifier: MIT
"""
Flask Dashboard + Inference API for vendor-neutral predictive maintenance.
Provides: training, model loading, prediction endpoint, and dashboard data APIs.
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import json
import os
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
model = None
scaler = None
data_history = []
servers_data = []

# Color scheme for charts
CHART_COLORS = {
    'primary': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'sequential': ['#FF9999', '#FF6666', '#FF3333', '#FF0000'],
    'categorical': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD', '#96CEB4']
}

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def generate_server_data(num_servers=50):
    """Generate simulated data for multiple servers"""
    servers = []
    np.random.seed(42)

    server_names = [
        f"WEB-{i:03d}" for i in range(1, 21)
    ] + [
        f"DB-{i:03d}" for i in range(1, 11)
    ] + [
        f"APP-{i:03d}" for i in range(1, 11)
    ] + [
        f"CACHE-{i:03d}" for i in range(1, 9)
    ]

    for i, name in enumerate(server_names[:num_servers]):
        server_type = name.split('-')[0]

        # Base values based on server type
        base_values = {
            'WEB': {'cpu': 1500000, 'memory': 80, 'fail_prob': 0.1},
            'DB': {'cpu': 1800000, 'memory': 90, 'fail_prob': 0.15},
            'APP': {'cpu': 1200000, 'memory': 70, 'fail_prob': 0.08},
            'CACHE': {'cpu': 1000000, 'memory': 60, 'fail_prob': 0.05}
        }

        base = base_values.get(server_type, {'cpu': 1200000, 'memory': 75, 'fail_prob': 0.1})

        # Simulate aging factor (older servers have higher failure probability)
        age_factor = np.random.uniform(0.8, 1.5)

        server = {
            'id': i,
            'name': name,
            'type': server_type,
            'status': np.random.choice(['Healthy', 'Warning', 'Critical'],
                                      p=[0.7, 0.2, 0.1]),
            'avg_cpu': int(base['cpu'] * np.random.uniform(0.8, 1.2)),
            'memory_usage': int(base['memory'] * np.random.uniform(0.7, 1.3)),
            'temperature': np.random.normal(65, 8),
            'uptime_days': int(np.random.exponential(30)),
            'last_maintenance': (datetime.now() - timedelta(
                days=np.random.randint(7, 180))).isoformat(),
            'failure_risk': min(100, base['fail_prob'] * age_factor * 100),
            'cluster': np.random.randint(0, 4)
        }

        # Ensure values are within bounds
        server['memory_usage'] = max(10, min(100, server['memory_usage']))
        server['temperature'] = max(30, min(100, server['temperature']))
        server['failure_risk'] = max(0, min(100, server['failure_risk']))

        servers.append(server)

    return servers

def generate_time_series_data(days=30):
    """Generate time series data for charts"""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')

    # Generate trends with seasonality
    base_trend = np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7))  # Weekly seasonality

    data = []
    for i, date in enumerate(dates):
        hour = date.hour

        # CPU usage (higher during business hours)
        cpu_base = 1200000 + (500000 if 9 <= hour <= 17 else 0)
        cpu = cpu_base + base_trend[i] * 300000 + np.random.normal(0, 100000)

        # Memory usage
        memory = 70 + base_trend[i] * 15 + np.random.normal(0, 5)

        # Temperature (correlated with CPU)
        temp = 60 + (cpu / 50000) + np.random.normal(0, 3)

        # Failure incidents (higher during high load)
        failures = 1 if (cpu > 1800000 and np.random.random() < 0.3) else 0

        # Network latency
        latency = 50 + (20 if 9 <= hour <= 17 else 0) + np.random.normal(0, 10)

        # Disk I/O
        disk_io = 500 + base_trend[i] * 200 + np.random.normal(0, 50)

        data.append({
            'timestamp': date.isoformat(),
            'avg_cpu': max(500000, min(2500000, cpu)),
            'memory_usage': max(20, min(100, memory)),
            'temperature': max(40, min(85, temp)),
            'failures': failures,
            'network_latency': max(10, min(200, latency)),
            'disk_io': max(100, min(1000, disk_io)),
            'hour': hour,
            'day_of_week': date.dayofweek,
            'is_weekend': 1 if date.dayofweek >= 5 else 0
        })

    return pd.DataFrame(data)

def train_predictive_model():
    """Train the main predictive model"""
    print("Training predictive maintenance model...")

    # Generate training data
    df = generate_time_series_data(90)  # 90 days of data

    # Create features for failure prediction in next 24 hours
    # Simulate failure labels (1 = failure in next 24 hours)
    df['failure_next_24h'] = 0

    # Create failure events based on thresholds
    high_risk_conditions = (
        (df['avg_cpu'] > 2000000) |
        (df['temperature'] > 75) |
        (df['memory_usage'] > 90)
    )

    # Mark next 24 hours as failures
    for i in range(len(df)):
        if high_risk_conditions.iloc[i]:
            for j in range(i+1, min(i+25, len(df))):
                df.loc[df.index[j], 'failure_next_24h'] = 1

    # Add some noise
    failure_indices = df[df['failure_next_24h'] == 1].index
    non_failure_indices = df[df['failure_next_24h'] == 0].index

    # Flip 5% of labels
    n_flip = int(len(failure_indices) * 0.05)
    if n_flip > 0:
        flip_idx = np.random.choice(failure_indices, n_flip, replace=False)
        df.loc[flip_idx, 'failure_next_24h'] = 0

    n_flip = int(len(non_failure_indices) * 0.02)
    if n_flip > 0:
        flip_idx = np.random.choice(non_failure_indices, n_flip, replace=False)
        df.loc[flip_idx, 'failure_next_24h'] = 1

    # Feature engineering
    df['cpu_to_temp_ratio'] = df['avg_cpu'] / df['temperature']
    df['memory_pressure'] = df['memory_usage'] * df['disk_io'] / 1000
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Features for prediction
    features = [
        'avg_cpu', 'memory_usage', 'temperature', 'network_latency',
        'disk_io', 'cpu_to_temp_ratio', 'memory_pressure',
        'hour_sin', 'hour_cos', 'is_weekend'
    ]

    X = df[features]
    y = df['failure_next_24h']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

    best_model = None
    best_score = 0
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'accuracy': float(accuracy),
            'model': model,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        print(f"{name} Accuracy: {accuracy:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = model

    # Save the best model and scaler
    if best_model is not None:
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'features': features,
            'training_date': datetime.now().isoformat(),
            'accuracy': float(best_score)
        }

        joblib.dump(model_data, 'models/predictive_model.pkl')
        print(f"\nBest model saved with accuracy: {best_score:.4f}")

        # Save metadata - FIXED JSON serialization
        class_dist = y.value_counts()
        metadata = {
            'model_type': type(best_model).__name__,
            'accuracy': float(best_score),
            'training_date': datetime.now().isoformat(),
            'features': features,
            'n_samples': int(len(df)),
            'class_distribution': {int(k): int(v) for k, v in class_dist.items()}
        }

        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4, cls=NumpyEncoder)

    return results

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Get all data needed for dashboard charts"""
    try:
        # 1. Server Status Distribution (Pie Chart)
        servers = generate_server_data(50)
        status_counts = {}
        for server in servers:
            status = server['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        # 2. Failure Risk by Server Type (Bar Chart)
        risk_by_type = {}
        for server in servers:
            server_type = server['type']
            if server_type not in risk_by_type:
                risk_by_type[server_type] = []
            risk_by_type[server_type].append(server['failure_risk'])

        avg_risk_by_type = {k: np.mean(v) for k, v in risk_by_type.items()}

        # 3. CPU Usage Over Time (Line Chart)
        time_data = generate_time_series_data(7)  # Last 7 days
        cpu_timeline = []
        for _, row in time_data.iterrows():
            cpu_timeline.append({
                'timestamp': row['timestamp'],
                'cpu': row['avg_cpu'] / 10000,  # Scale for display
                'memory': row['memory_usage']
            })

        # 4. Temperature Distribution (Histogram)
        temperatures = [s['temperature'] for s in servers]

        # 5. Maintenance Schedule (Gantt-like bar chart)
        maintenance_data = []
        for server in servers[:10]:  # Show first 10 servers
            last_maintenance = datetime.fromisoformat(server['last_maintenance'])
            days_since = (datetime.now() - last_maintenance).days
            next_due = min(100, (days_since / 180) * 100)  # Scale to percentage

            maintenance_data.append({
                'server': server['name'],
                'days_since': days_since,
                'next_due_percent': next_due,
                'status': server['status']
            })

        # 6. Failure Prediction Confidence (Radar/Donut)
        prediction_confidence = {
            'High Confidence (>90%)': len([s for s in servers if s['failure_risk'] > 90]),
            'Medium Confidence (70-90%)': len([s for s in servers if 70 <= s['failure_risk'] <= 90]),
            'Low Confidence (<70%)': len([s for s in servers if s['failure_risk'] < 70])
        }

        # Overall statistics
        total_servers = len(servers)
        critical_servers = len([s for s in servers if s['status'] == 'Critical'])
        avg_failure_risk = np.mean([s['failure_risk'] for s in servers])
        avg_cpu = np.mean([s['avg_cpu'] for s in servers])
        avg_temp = np.mean([s['temperature'] for s in servers])

        # Simulate current incidents
        current_incidents = [
            {
                'id': 1,
                'server': 'DB-005',
                'type': 'High CPU',
                'severity': 'Critical',
                'time': '5 minutes ago',
                'description': 'CPU usage above 95% for 15 minutes'
            },
            {
                'id': 2,
                'server': 'WEB-012',
                'type': 'Memory Leak',
                'severity': 'Warning',
                'time': '12 minutes ago',
                'description': 'Memory usage increasing steadily'
            }
        ]

        return jsonify({
            # Chart Data
            'server_status': status_counts,
            'risk_by_type': avg_risk_by_type,
            'cpu_timeline': cpu_timeline[-24:],  # Last 24 hours
            'temperatures': temperatures,
            'maintenance_schedule': maintenance_data,
            'prediction_confidence': prediction_confidence,

            # Statistics
            'total_servers': total_servers,
            'critical_servers': critical_servers,
            'avg_failure_risk': float(avg_failure_risk),
            'avg_cpu': float(avg_cpu),
            'avg_temp': float(avg_temp),
            'current_incidents': current_incidents,

            # Server list for table
            'servers': servers[:15],  # First 15 servers for table

            # Time series for detailed charts
            'time_series_7d': time_data.to_dict('records'),

            # Colors for charts
            'colors': CHART_COLORS
        })
    except Exception as e:
        app.logger.error(f"Error in dashboard-data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-failure', methods=['POST'])
def predict_failure():
    """Predict failure for specific server metrics"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        print(f"Prediction request received: {data}")

        # Load model if not loaded
        global model, scaler
        if model is None or scaler is None:
            try:
                model_data = joblib.load('models/predictive_model.pkl')
                model = model_data['model']
                scaler = model_data['scaler']
                print("Model loaded successfully for prediction")
            except Exception as e:
                print(f"Error loading model: {e}")
                return jsonify({
                    'error': 'Model not available. Please train first.',
                    'details': str(e)
                }), 400

        try:
            # Prepare features with defaults - CONVERT TO APPROPRIATE TYPES
            current_hour = datetime.now().hour
            
            # Convert all values to proper numeric types
            avg_cpu = float(data.get('avg_cpu', 1200000))
            memory_usage = float(data.get('memory_usage', 75))
            temperature = float(data.get('temperature', 65))
            network_latency = float(data.get('network_latency', 50))
            disk_io = float(data.get('disk_io', 500))
            hour = int(data.get('hour', current_hour))

            print(f"Using features - CPU: {avg_cpu}, Mem: {memory_usage}, Temp: {temperature}")

            # Calculate derived features
            cpu_to_temp_ratio = avg_cpu / max(1, temperature)
            memory_pressure = memory_usage * disk_io / 1000
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            is_weekend = 1 if datetime.now().weekday() >= 5 else 0

            features = np.array([[
                avg_cpu,
                memory_usage,
                temperature,
                network_latency,
                disk_io,
                cpu_to_temp_ratio,
                memory_pressure,
                hour_sin,
                hour_cos,
                is_weekend
            ]])

            print(f"Features prepared: {features}")

            # Scale features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0][1])

            print(f"Prediction: {prediction}, Probability: {probability}")

            # Determine risk level
            if probability > 0.8:
                risk_level = 'CRITICAL'
                recommendation = 'Immediate maintenance required. Schedule downtime.'
            elif probability > 0.5:
                risk_level = 'HIGH'
                recommendation = 'Schedule maintenance within 24 hours.'
            elif probability > 0.2:
                risk_level = 'MEDIUM'
                recommendation = 'Monitor closely. Schedule maintenance this week.'
            else:
                risk_level = 'LOW'
                recommendation = 'Continue normal operations.'

            return jsonify({
                'success': True,
                'failure_predicted': bool(prediction),
                'confidence': probability,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'estimated_time_to_failure': f"{int((1-probability)*48)} hours" if probability > 0.3 else "> 48 hours",
                'features_used': {
                    'avg_cpu': avg_cpu,
                    'memory_usage': memory_usage,
                    'temperature': temperature
                }
            })

        except Exception as e:
            print(f"Error during prediction calculation: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

    except Exception as e:
        print(f"General error in predict_failure: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train or retrain the model"""
    try:
        print("Starting model training via API...")
        results = train_predictive_model()
        print(f"Training completed, got results: {bool(results)}")

        # Load the newly trained model
        global model, scaler
        try:
            model_data = joblib.load('models/predictive_model.pkl')
            model = model_data['model']
            scaler = model_data['scaler']
            print("Model loaded successfully after training")
        except Exception as load_error:
            print(f"Error loading model after training: {load_error}")
            # Try to get model from results
            if results and len(results) > 0:
                first_model_name = list(results.keys())[0]
                model = results[first_model_name]['model']
                print(f"Using model from training results: {first_model_name}")
            else:
                return jsonify({
                    'success': False,
                    'error': 'Model trained but could not be loaded',
                    'details': str(load_error)
                }), 500

        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': float(results[list(results.keys())[0]]['accuracy']) if results else 0.0,
            'model_type': type(model).__name__ if model else 'Unknown'
        })
    except Exception as e:
        print(f"Error in train_model API: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-report')
def generate_report():
    """Generate a visual report as an image"""
    try:
        # Generate sample data for report
        servers = generate_server_data(20)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Predictive Maintenance Report', fontsize=16, fontweight='bold')

        # Plot 1: Server Status Distribution
        status_counts = {}
        for server in servers:
            status_counts[server['status']] = status_counts.get(server['status'], 0) + 1

        axes[0, 0].pie(status_counts.values(), labels=status_counts.keys(),
                      autopct='%1.1f%%', colors=CHART_COLORS['primary'][:3])
        axes[0, 0].set_title('Server Status Distribution')

        # Plot 2: CPU Usage by Server Type
        cpu_by_type = {}
        for server in servers:
            cpu_by_type[server['type']] = cpu_by_type.get(server['type'], []) + [server['avg_cpu']]

        avg_cpu_by_type = {k: np.mean(v) for k, v in cpu_by_type.items()}
        axes[0, 1].bar(avg_cpu_by_type.keys(), [v/10000 for v in avg_cpu_by_type.values()],
                      color=CHART_COLORS['primary'])
        axes[0, 1].set_title('Average CPU Usage by Server Type')
        axes[0, 1].set_ylabel('CPU (scaled)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Temperature Distribution
        temps = [s['temperature'] for s in servers]
        axes[1, 0].hist(temps, bins=10, color=CHART_COLORS['sequential'][0], edgecolor='black')
        axes[1, 0].set_title('Server Temperature Distribution')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Count')

        # Plot 4: Failure Risk by Server
        risks = [s['failure_risk'] for s in servers]
        server_names = [s['name'] for s in servers]

        # Sort by risk for better visualization
        sorted_indices = np.argsort(risks)[-10:]  # Top 10 highest risk
        sorted_risks = [risks[i] for i in sorted_indices]
        sorted_names = [server_names[i] for i in sorted_indices]

        colors = [CHART_COLORS['sequential'][3] if r > 80 else
                 CHART_COLORS['sequential'][2] if r > 60 else
                 CHART_COLORS['sequential'][1] for r in sorted_risks]

        axes[1, 1].barh(sorted_names, sorted_risks, color=colors)
        axes[1, 1].set_title('Top 10 High-Risk Servers')
        axes[1, 1].set_xlabel('Failure Risk (%)')

        plt.tight_layout()

        # Save plot to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        # Encode image to base64
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'image': f'data:image/png;base64,{img_str}',
            'report_date': datetime.now().isoformat(),
            'total_servers': len(servers),
            'high_risk_count': len([s for s in servers if s['failure_risk'] > 80])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/real-time-metrics')
def get_real_time_metrics():
    """Get real-time metrics stream"""
    current_time = datetime.now()

    # Generate current metrics
    metrics = {
        'timestamp': current_time.isoformat(),
        'total_requests': int(np.random.poisson(1000)),
        'avg_response_time': float(np.random.normal(150, 20)),
        'error_rate': float(np.random.uniform(0.1, 2.5)),
        'active_connections': int(np.random.randint(500, 2000)),
        'data_throughput': float(np.random.normal(500, 50)),  # MB/s
        'system_load': float(np.random.uniform(0.3, 0.9))
    }

    # Add to history
    data_history.append(metrics)
    if len(data_history) > 1000:
        data_history.pop(0)

    return jsonify(metrics)

def load_model():
    """Load the trained model"""
    global model, scaler
    try:
        if os.path.exists('models/predictive_model.pkl'):
            model_data = joblib.load('models/predictive_model.pkl')
            model = model_data['model']
            scaler = model_data['scaler']
            print("Model loaded successfully on startup")
        else:
            print("No pre-trained model found. Will train on first run.")
            model = None
            scaler = None
    except Exception as e:
        print(f"Error loading model on startup: {e}")
        model = None
        scaler = None

if __name__ == '__main__':
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Set style for plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Train initial model if not exists
    if not os.path.exists('models/predictive_model.pkl'):
        print("Training initial model...")
        try:
            train_predictive_model()
        except Exception as e:
            print(f"Initial training failed: {e}")
            print("Creating dummy model for testing...")
            # Create a simple dummy model
            from sklearn.dummy import DummyClassifier
            import numpy as np

            dummy_model = DummyClassifier(strategy='stratified')
            X_dummy = np.random.randn(100, 10)
            y_dummy = np.random.randint(0, 2, 100)
            dummy_model.fit(X_dummy, y_dummy)

            model_data = {
                'model': dummy_model,
                'scaler': StandardScaler(),
                'features': [f'feature_{i}' for i in range(10)],
                'training_date': datetime.now().isoformat(),
                'accuracy': 0.5
            }

            joblib.dump(model_data, 'models/predictive_model.pkl')
            print("Dummy model created for testing")

    # Load the model
    load_model()

    print("\n" + "="*60)
    print("Predictive Maintenance Dashboard")
    print("="*60)
    print("Dashboard URL: http://localhost:5000")
    print("API Documentation:")
    print("  GET  /                    - Main Dashboard")
    print("  GET  /api/dashboard-data  - Chart data")
    print("  POST /api/predict-failure - Make prediction")
    print("  POST /api/train-model     - Retrain model")
    print("  GET  /api/generate-report - Generate visual report")
    print("  GET  /api/real-time-metrics - Real-time metrics")
    print("="*60 + "\n")

    # Run the app
