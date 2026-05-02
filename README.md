<div align="center">

# 🧠 Fault-Tolerant MLOps Pipeline

### Zero-Shot Learning · Self-Healing · Real-Time Observability

[
[
[
[
[
[

***

*A production-grade, self-healing ML inference pipeline that uses Zero-Shot Learning to detect unseen fault types in real time — without retraining.*

</div>

***

## 📸 Dashboard Preview

> Live Grafana dashboard showing ZSL fault detection, MTTR, availability, and incident tracking in real time.



***

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Traffic Layer                               │
│   load_gen.py  ──►  FastAPI App (port 8000)  ──►  /predict         │
│                         │                                           │
│                         ▼                                           │
│              app_requests_total (Counter)                           │
│              app_prediction_errors_total (Counter)                  │
│              app_request_latency_seconds (Histogram)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Observability Layer                            │
│                                                                     │
│   Prometheus (port 9090)  ◄──  scrapes /metrics every 5s           │
│         │                      from app + zsl + cadvisor            │
│         │                                                           │
│         ▼                                                           │
│   cAdvisor  ──►  container_cpu / mem / net / fs metrics             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ZSL Inference Layer (port 8001)                   │
│                                                                     │
│   ETL Feeder Thread                                                 │
│   ├── Polls Prometheus every 5s                                     │
│   ├── Builds 14-feature vector (API + container metrics)            │
│   ├── Normalizes via fitted StandardScaler                          │
│   ├── Embeds via MLP Encoder → 16-dim embedding space               │
│   ├── Cosine similarity vs class centroids                          │
│   └── Labels: normal / error_spike / slow / UNSEEN                 │
│                                                                     │
│   MTTR Tracker                                                      │
│   ├── Records fault onset timestamp                                 │
│   ├── Measures time-to-recovery                                     │
│   └── Logs closed incidents                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Controller Layer (port 8002)                     │
│                                                                     │
│   Polls ZSL /incidents + /health every 5s                          │
│   Exposes:                                                          │
│   ├── controller_last_mttr_seconds                                  │
│   ├── controller_availability                                       │
│   ├── controller_active_incident                                    │
│   └── controller_incidents_closed_total                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Visualization Layer                            │
│                                                                     │
│   Grafana (port 3000)  ──►  ZSL MLOps Dashboard                    │
│   ├── Request Rate & P95 Latency (real-time)                        │
│   ├── ZSL Confidence + UNSEEN Streak                                │
│   ├── Controller MTTR + Availability %                              │
│   └── Active Incident + Closed Incidents                            │
└─────────────────────────────────────────────────────────────────────┘
```

***

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Zero-Shot Fault Detection** | Detects fault types *never seen during training* using cosine similarity in embedding space |
| ⚡ **Real-Time ETL Pipeline** | 14-dimensional feature vector built from live Prometheus metrics every 5 seconds |
| 🧬 **MLP Embedding Model** | Trained on synthetic data; 16-dim embedding space with class centroids for similarity scoring |
| 🚨 **MTTR Tracking** | Automatic fault onset/recovery detection with Mean Time To Recovery measurement |
| 📊 **Full Observability** | Prometheus + Grafana + cAdvisor for API, model, and container-level metrics |
| 🔁 **Fault Injection API** | `/setmode/error_spike`, `/setmode/slow`, `/setmode/normal` for live demo scenarios |
| 🛡️ **Self-Healing Design** | Controller service aggregates health signals and can trigger automated remediation |
| 🐳 **Fully Containerized** | One `docker compose up` deploys the entire stack — no manual setup |

***

## 🧪 The ZSL Model

The core innovation is a **Zero-Shot Learning classifier** that can identify fault conditions it was never explicitly trained on.

### How It Works

1. **Training**: A synthetic log generator produces labeled windows for 5 fault types (`normal`, `error_spike`, `slow`, `memory_leak`, `intermittent`). A 3-class MLP encoder is trained on `normal`, `error_spike`, and `slow`.

2. **Embedding Space**: The trained encoder maps 14-dimensional feature vectors into a 16-dimensional embedding space. Class centroids are computed from validation embeddings.

3. **Inference**: At runtime, live metrics are embedded and compared to centroids via cosine similarity. If the best similarity score falls below the threshold (5th percentile of validation similarities), the window is labeled `UNSEEN` — a zero-shot detection of a novel fault.

4. **UNSEEN Detection**: `memory_leak` and `intermittent` were *not* trained as explicit classes — they are detected as `UNSEEN` when encountered in production, triggering a retraining recommendation.

### Feature Vector (14 dimensions)

```
API Metrics (7):
  [0] mean_latency       — avg response time (s)
  [1] max_latency        — p99 latency (s)
  [2] p95_latency        — 95th percentile latency (s)
  [3] std_latency        — latency variance
  [4] error_rate         — fraction of 500 responses
  [5] request_rate       — req/s
  [6] http_500_count     — raw error count in window

Container Metrics (7, via cAdvisor):
  [7]  cpu_percent       — CPU usage %
  [8]  mem_usage_mb      — memory RSS (MB)
  [9]  mem_percent       — memory utilization %
  [10] io_read_bytes     — disk read rate (B/s)
  [11] io_write_bytes    — disk write rate (B/s)
  [12] net_rx_bytes      — network receive (B/s)
  [13] net_tx_bytes      — network transmit (B/s)
```

### Model Architecture

```python
MLPEncoder(
  encoder: Linear(14→64) → ReLU → Dropout(0.2)
         → Linear(64→32) → ReLU → Dropout(0.2)
         → Linear(32→16)          # 16-dim embedding
  classifier: Linear(16→3)        # for training only
)
```

Training reaches `val_acc=1.000` within 150 epochs with cosine similarity threshold auto-calibrated at the 5th validation percentile.

***

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (with Compose v2)
- Python 3.11+ (for running load generator locally)
- 4 GB RAM minimum

### 1. Clone & Launch

```bash
git clone https://github.com/dan0003umd/fault-tolerant-mlops.git
cd fault-tolerant-mlops
docker compose up --build -d
```

> The ZSL container automatically runs `generate_synthetic_log.py` and `train.py` at build time. First build takes ~3 minutes.

### 2. Verify Everything Is Up

```bash
docker compose ps
```

| Service | Port | Purpose |
|---------|------|---------|
| `app` | `8000` | FastAPI ML inference app |
| `zsl` | `8001` | ZSL fault classifier + ETL |
| `controller` | `8002` | MTTR & availability aggregator |
| `prometheus` | `9090` | Metrics scraper |
| `grafana` | `3000` | Dashboard UI |
| `cadvisor` | `8080` | Container metrics |

### 3. Open Grafana Dashboard

Navigate to **[http://localhost:3000](http://localhost:3000)**

- Username: `admin` / Password: `admin`
- Dashboard: **ZSL MLOps → Self-Healing Pipeline**

### 4. Start Load Generator

```powershell
# Windows PowerShell — continuous traffic at 10 RPS
while ($true) {
    python load_generator/load_gen.py
    Start-Sleep -Milliseconds 100
}
```

```bash
# Linux / macOS
while true; do python load_generator/load_gen.py; sleep 0.1; done
```

### 5. Inject a Fault

```powershell
# Trigger error spike (30% of requests return 500)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/setmode/error_spike"

# Slow mode (high latency)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/setmode/slow"

# Return to normal
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/setmode/normal"
```

Within **15–30 seconds**, the ZSL dashboard will show the label flip from `normal` to `error_spike` with confidence > 0.90.

***

## 📡 API Reference

### ML App (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | `POST` | Run inference on input payload |
| `/health` | `GET` | App health check |
| `/metrics` | `GET` | Prometheus metrics |
| `/setmode/{mode}` | `POST` | Inject fault: `normal`, `error_spike`, `slow` |

### ZSL Server (port 8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | `POST` | Classify a 14-feature window |
| `/latest` | `GET` | Most recent ETL classification result |
| `/incidents` | `GET` | Full incident log with MTTR history |
| `/health` | `GET` | ZSL health + active fault status |
| `/metrics` | `GET` | Prometheus metrics (confidence, streak, MTTR) |

### Example: Check Latest Classification

```bash
curl http://localhost:8001/latest
```

```json
{
  "label": "error_spike",
  "confidence": 0.9612,
  "unseen_flag": false,
  "closest_known": "error_spike",
  "unseen_streak": 0,
  "retrain_recommended": false,
  "similarities": {
    "normal": -0.7664,
    "error_spike": 0.9612,
    "slow": -0.0068
  },
  "threshold": 0.75
}
```

***

## 📊 Grafana Dashboard Panels

| Panel | Query | What It Shows |
|-------|-------|---------------|
| **Request Rate** | `sum(rate(app_requests_total[30s]))` | Live RPS from load generator |
| **P95 Latency** | `histogram_quantile(0.95, ...)` | 95th percentile response time |
| **Error Rate** | `sum(increase(errors)) / sum(increase(requests))` | Fraction of 500 responses |
| **ZSL Confidence** | `zsl_last_confidence` | Model confidence in current label |
| **UNSEEN Streak** | `zsl_unseen_streak` | Consecutive UNSEEN windows |
| **Controller MTTR** | `controller_last_mttr_seconds` | Time from fault onset to recovery |
| **Availability %** | `controller_availability * 100` | Rolling uptime percentage |
| **Active Incident** | `controller_active_incident` | 1 = fault active, 0 = healthy |
| **Closed Incidents** | `controller_incidents_closed_total` | Total resolved incidents |
| **ZSL Classify Latency** | `histogram_quantile(0.95, zsl_detection_latency_seconds_bucket)` | Inference pipeline latency |

***

## 📁 Project Structure

```
fault-tolerant-mlops/
│
├── main.py                      # FastAPI ML app with fault injection
├── zsl_server.py                # ZSL inference server + ETL feeder
├── controller.py                # MTTR/availability aggregator service
├── generate_synthetic_log.py    # Synthetic training data generator
├── train.py                     # MLP encoder training script
├── evaluate.py                  # Model evaluation & centroid export
│
├── load_generator/
│   └── load_gen.py              # HTTP load generator (10 RPS default)
│
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   └── mlops.json       # Grafana dashboard definition
│       └── datasources/
│           └── prometheus.yml   # Auto-provisioned data source
│
├── prometheus.yml               # Prometheus scrape config
├── docker-compose.yml           # Full stack orchestration
├── Dockerfile                   # ZSL container (build + train)
└── requirements.txt             # Python dependencies
```

***

## ⚙️ Configuration

All settings are controlled via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_URL` | `http://prometheus:9090` | Prometheus endpoint for ETL queries |
| `ETL_INTERVAL` | `5` | Seconds between ZSL inference cycles |
| `POLL_INTERVAL` | `5` | Controller polling interval (seconds) |
| `ZSL_URL` | `http://zsl:8001` | ZSL server URL for controller |

***

## 🔬 How to Run the Experiment (Demo Flow)

This is the recommended sequence for a live demonstration:

```
1. docker compose up --build -d          # Start stack
2. Start load generator (10 RPS)         # Baseline traffic
3. Watch Grafana — all panels stable     # Normal: conf ≈ 0.90, label=normal
4. POST /setmode/error_spike             # Inject fault
5. Wait 15–30 seconds                    # ETL detects → label=error_spike
6. Watch UNSEEN Streak, MTTR panels      # Active incident = 1
7. POST /setmode/normal                  # Recover
8. Watch Availability drop then recover  # MTTR logged, incident closed
```

***

## 🧑‍🔬 Research Context

This project was built as part of coursework in **Applied Machine Learning (MSML 605)** at the **University of Maryland**. It demonstrates:

- **Zero-Shot Learning** applied to operational fault detection in MLOps pipelines
- **Self-healing infrastructure** where anomaly detection drives automated response
- **Production observability** patterns using Prometheus and Grafana
- **Fault-tolerant design** with MTTR measurement, incident tracking, and uptime availability metrics

The core contribution is showing that a ZSL model trained on 3 labeled fault classes can generalize to detect 2 unseen fault types (`memory_leak`, `intermittent`) via cosine similarity threshold in embedding space — without retraining.

***

## 🤝 Contributing

Pull requests welcome! Key areas for improvement:

- [ ] Automated remediation actions triggered by `controller_active_incident`
- [ ] Sliding window retraining pipeline when UNSEEN streak exceeds threshold
- [ ] Alert manager integration (PagerDuty / Slack webhooks)
- [ ] Kubernetes deployment manifests (Helm chart)
- [ ] A/B model versioning with traffic splitting

***

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

***

<div align="center">

Built with ⚡ by [@dan0003umd](https://github.com/dan0003umd)

*Zero-shot. Self-healing. Production-ready.*

</div>
