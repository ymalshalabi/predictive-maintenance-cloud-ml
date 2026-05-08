#  Following phase-1, I have worked on implementing

rolling windows, real telemetry used from : https://github.com/alejandrofdez-us/DataCenter-Traces-Datasets , 
using MLP with Random Forest and improving dashboard.


## Technical Architecture
The system is built upon a 6-layer microservices architecture, fully orchestrated via Docker Compose to ensure environmental consistency across AWS, OpenStack, or local clusters.

Layer 1: Data Acquisition – Utilizing the Azure Public Dataset v2, re-sampled at 10-second intervals to capture high-resolution system volatility

Layer 2: Ingestion – A distributed 3-node cluster architecture designed for high-availability telemetry streaming

Layer 3: Transport – A RESTful API facade developed with Flask, utilizing gevent for nonblocking I/O performance.

Layer 4: Processing & Inference – A dual-model inference engine featuring a Multi-Layer Perceptron (MLP) for anomaly detection and Random Forest for baseline comparison.

Layer 5: Storage – High-concurrency persistence using PostgreSQL for telemetry history and Redis for real-time caching.

Layer 6: Visualization – A real-time monitoring dashboard providing sub-400ms end-to-end latency for critical system metrics.

## Repository Structure
/dashboard: Contains the visualization templates for real-time monitoring.

/telemetry-api: Source code for the distributed ingestion and heartbeat logic.

/docker: Orchestration files including Dockerfile and docker-compose configurations.

mlp_anomaly_model.pkl: Serialized Multi-Layer Perceptron model optimized for reconstruction error analysis.

EDAv2.ipynb: Comprehensive Exploratory Data Analysis of the 10-second grouped telemetry dataset.

train_model.py: Training pipeline for model retraining and validation.

## Performance Metrics
Ingestion Granularity: 10-second telemetry windows
Dashboard Latency: Verified sub-400ms end-to-end response time
High-fidelity anomaly detection with minimized reconstruction error on unseen cloud noise.

## Deployment Instructions
Ensure Docker and Docker Compose are installed on the host machine or EC2 instance

1- clone the repository:
git clone -b phase-2 https://github.com/ymalshalabi/predictive-maintenance-cloud-ml.git

2- Launch the 6-layer stack by using docker-compose up -d

3- Access the monitoring interface at http://localhost:5000

## Contributors: 
### Dr Ayaz Ul Hassan Khan
### Yasmin Alshalabi 
### Academic Institution: King Fahd University of Petroleum and Minerals (KFUPM)




Dashboard Latency: Verified sub-400ms end-to-end response time.

Model Accuracy: High-fidelity anomaly detection with minimized reconstruction error on unseen cloud noise.
