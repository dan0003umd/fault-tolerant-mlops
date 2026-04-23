"""
ZSL Fault Diagnosis Server
---------------------------
Real-time fault classification service for the MLOps pipeline.

Features:
  - Polls /predict endpoint and Docker stats on a configurable interval
  - Builds feature windows and classifies via ZSL model
  - Tracks UNSEEN streaks for event-triggered model updates
  - Logs all classifications to zsl_classifications.log
  - REST API for Soumitra's self-healing controller

Endpoints:
  GET  /status           — server health + model info
  GET  /latest           — most recent classification result
  GET  /history          — last N classification results
  POST /classify         — classify a manually provided feature vector
  GET  /retrain          — reload model from disk after external retrain
  GET  /start_monitor    — start background monitoring loop
  GET  /stop_monitor     — stop background monitoring loop

Usage:
  python3 zsl_server.py --port 8001
  python3 zsl_server.py --port 8001 --app-url http://localhost:8000
"""

import json
import time
import asyncio
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import deque

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
import torch.nn as nn
import uvicorn
import requests as http_requests

from zsl_model import FEATURE_COLS, KNOWN_CLASSES, zsl_classify
from sklearn.metrics.pairwise import cosine_similarity

# ---- Logging setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zsl_server")

app = FastAPI(
    title="ZSL Fault Diagnosis Server",
    description="Real-time zero-shot fault classification for fault-tolerant MLOps",
    version="1.0.0",
)

# ---- Global state ----
MODEL = None
SCALER = None
LABEL_ENC = None
CENTROIDS = None
THRESHOLD = None
UNSEEN_STREAK = 0
CLASSIFICATION_LOG = deque(maxlen=1000)  # rolling buffer of last 1000 results
MONITOR_TASK = None
MONITOR_RUNNING = False
REQUEST_BUFFER = deque(maxlen=500)  # buffer of raw requests for windowing
APP_URL = "http://localhost:8000"
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "10"))  # seconds between classifications
LOG_FILE = "zsl_classifications.log"
AUTO_START_MONITOR = os.getenv("AUTO_START_MONITOR", "false").lower() == "true"
APP_URL = os.getenv("APP_URL", APP_URL)


# ---- Request/Response models ----

class FeatureVector(BaseModel):
    mean_latency: float
    max_latency: float
    p95_latency: float
    std_latency: float
    error_rate: float
    request_rate: float
    http_500_count: float
    cpu_percent: float = 0.0
    mem_usage_mb: float = 0.0
    mem_percent: float = 0.0
    io_read_bytes: float = 0.0
    io_write_bytes: float = 0.0
    net_rx_bytes: float = 0.0
    net_tx_bytes: float = 0.0


class ClassificationResult(BaseModel):
    timestamp: str
    prediction: str
    confidence: float
    unseen_streak: int
    retrain_recommended: bool
    features: dict


# ---- Model loading ----

def load_torch_model(model_dir):
    """Load PyTorch embedding model."""
    pt_path = model_dir / "embedding_model.pt"
    if not pt_path.exists():
        return None

    label_enc = joblib.load(model_dir / "label_encoder.joblib")
    n_classes = len(label_enc.classes_)
    input_dim = len(FEATURE_COLS)

    class EmbeddingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 16),
            )
            self.classifier = nn.Linear(16, n_classes)

        def get_embedding(self, x):
            return self.encoder(x)

        def forward(self, x):
            return self.classifier(self.encoder(x))

    model = EmbeddingMLP()
    model.load_state_dict(torch.load(pt_path, weights_only=True))
    model.eval()

    class TorchWrapper:
        def __init__(self, m):
            self.model = m
            self.encoder = m.encoder

        def get_embedding(self, X):
            self.model.eval()
            with torch.no_grad():
                return self.model.get_embedding(torch.FloatTensor(X)).numpy()

    return TorchWrapper(model)


def load_model(model_dir="zsl_output"):
    """Load all model artifacts."""
    global MODEL, SCALER, LABEL_ENC, CENTROIDS, THRESHOLD
    model_dir = Path(model_dir)

    SCALER = joblib.load(model_dir / "scaler.joblib")
    LABEL_ENC = joblib.load(model_dir / "label_encoder.joblib")

    with open(model_dir / "centroids.json") as f:
        cdata = json.load(f)
    THRESHOLD = cdata.pop("__threshold__")
    CENTROIDS = {k: np.array(v) for k, v in cdata.items()}

    # Try torch first, fall back to sklearn
    MODEL = load_torch_model(model_dir)
    if MODEL is None:
        MODEL = joblib.load(model_dir / "embedding_model.joblib")
        logger.info("Loaded sklearn model")
    else:
        logger.info("Loaded PyTorch model")


# ---- Classification logic ----

def classify_features(feature_dict):
    """Classify a single feature vector and update state."""
    global UNSEEN_STREAK

    feature_vec = [feature_dict.get(col, 0.0) for col in FEATURE_COLS]
    X = SCALER.transform([feature_vec])
    predictions, confidences = zsl_classify(MODEL, X, CENTROIDS, threshold=THRESHOLD)

    prediction = predictions[0]
    confidence = confidences[0]

    if prediction == "UNSEEN":
        UNSEEN_STREAK += 1
    else:
        UNSEEN_STREAK = 0

    retrain_recommended = UNSEEN_STREAK >= 5

    result = ClassificationResult(
        timestamp=datetime.utcnow().isoformat(),
        prediction=prediction,
        confidence=round(confidence, 4),
        unseen_streak=UNSEEN_STREAK,
        retrain_recommended=retrain_recommended,
        features=feature_dict,
    )

    # Log to memory and file
    CLASSIFICATION_LOG.append(result.dict())
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result.dict()) + "\n")

    if retrain_recommended:
        logger.warning(f"RETRAIN RECOMMENDED: {UNSEEN_STREAK} consecutive UNSEEN detections")

    return result


# ---- Background monitoring ----

def poll_app_metrics(app_url, n_requests=10):
    """Send test requests to the app and collect latency/status data."""
    results = []
    for _ in range(n_requests):
        try:
            start = time.time()
            resp = http_requests.post(
                f"{app_url}/predict",
                json={"features": [5.1, 3.5, 1.4, 0.2]},
                timeout=5,
            )
            latency = time.time() - start
            results.append({"latency_sec": latency, "http_status": resp.status_code})
        except Exception:
            results.append({"latency_sec": 5.0, "http_status": 503})
    return results


def poll_container_metrics():
    """Get container CPU/memory stats via Docker API."""
    try:
        import docker
        client = docker.from_env()
        container = client.containers.get("ftmlops-app")
        stats = container.stats(stream=False)

        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                    stats["precpu_stats"]["cpu_usage"]["total_usage"]
        sys_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                    stats["precpu_stats"]["system_cpu_usage"]
        n_cpus = stats["cpu_stats"].get("online_cpus", 1)
        cpu_pct = (cpu_delta / sys_delta) * n_cpus * 100 if sys_delta > 0 else 0

        mem = stats["memory_stats"].get("usage", 0)
        mem_limit = stats["memory_stats"].get("limit", 1)

        return {
            "cpu_percent": round(cpu_pct, 4),
            "mem_usage_mb": round(mem / (1024 * 1024), 2),
            "mem_percent": round(mem / mem_limit * 100, 4),
        }
    except Exception:
        return {"cpu_percent": 0, "mem_usage_mb": 0, "mem_percent": 0}


def build_feature_window(request_results, container_stats):
    """Build a feature vector from collected metrics."""
    latencies = [r["latency_sec"] for r in request_results]
    statuses = [r["http_status"] for r in request_results]

    return {
        "mean_latency": float(np.mean(latencies)),
        "max_latency": float(np.max(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "std_latency": float(np.std(latencies)),
        "error_rate": float(np.mean([s != 200 for s in statuses])),
        "request_rate": len(request_results) / MONITOR_INTERVAL,
        "http_500_count": sum(1 for s in statuses if s == 500),
        "cpu_percent": container_stats.get("cpu_percent", 0),
        "mem_usage_mb": container_stats.get("mem_usage_mb", 0),
        "mem_percent": container_stats.get("mem_percent", 0),
        "io_read_bytes": 0,
        "io_write_bytes": 0,
        "net_rx_bytes": 0,
        "net_tx_bytes": 0,
    }


async def monitor_loop():
    """Background loop: poll metrics, classify, log."""
    global MONITOR_RUNNING
    logger.info(f"Monitor started (interval={MONITOR_INTERVAL}s, app={APP_URL})")

    while MONITOR_RUNNING:
        try:
            # Collect metrics
            request_results = poll_app_metrics(APP_URL)
            container_stats = poll_container_metrics()
            features = build_feature_window(request_results, container_stats)

            # Classify
            result = classify_features(features)
            logger.info(
                f"Classification: {result.prediction} "
                f"(confidence={result.confidence:.4f}, streak={result.unseen_streak})"
            )
        except Exception as e:
            logger.error(f"Monitor error: {e}")

        await asyncio.sleep(MONITOR_INTERVAL)


# ---- API Endpoints ----

@app.on_event("startup")
async def startup():
    global MONITOR_RUNNING, MONITOR_TASK
    load_model()
    logger.info("ZSL Fault Diagnosis Server ready")
    if AUTO_START_MONITOR and not MONITOR_RUNNING:
        MONITOR_RUNNING = True
        MONITOR_TASK = asyncio.create_task(monitor_loop())
        logger.info("ZSL monitor auto-started")


@app.get("/status")
def status():
    """Server health and model info."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "known_classes": KNOWN_CLASSES,
        "threshold": THRESHOLD,
        "unseen_streak": UNSEEN_STREAK,
        "monitor_running": MONITOR_RUNNING,
        "total_classifications": len(CLASSIFICATION_LOG),
        "app_url": APP_URL,
    }


@app.post("/classify")
def classify_endpoint(req: FeatureVector):
    """Classify a manually provided feature vector."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    feature_dict = req.dict()
    result = classify_features(feature_dict)
    return result.dict()


@app.get("/latest")
def latest():
    """Get most recent classification result."""
    if not CLASSIFICATION_LOG:
        return {"message": "No classifications yet"}
    return CLASSIFICATION_LOG[-1]


@app.get("/history")
def history(n: int = 20):
    """Get last N classification results."""
    items = list(CLASSIFICATION_LOG)[-n:]
    return {
        "count": len(items),
        "total": len(CLASSIFICATION_LOG),
        "results": items,
    }


@app.get("/start_monitor")
async def start_monitor():
    """Start background monitoring loop."""
    global MONITOR_TASK, MONITOR_RUNNING
    if MONITOR_RUNNING:
        return {"message": "Monitor already running"}
    MONITOR_RUNNING = True
    MONITOR_TASK = asyncio.create_task(monitor_loop())
    return {"message": "Monitor started", "interval": MONITOR_INTERVAL}


@app.get("/stop_monitor")
async def stop_monitor():
    """Stop background monitoring loop."""
    global MONITOR_RUNNING
    MONITOR_RUNNING = False
    return {"message": "Monitor stopped"}


@app.get("/retrain")
def retrain():
    """Reload model from disk after external retrain."""
    global UNSEEN_STREAK
    load_model()
    UNSEEN_STREAK = 0
    return {"status": "model reloaded", "unseen_streak_reset": True}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--app-url", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    APP_URL = args.app_url
    MONITOR_INTERVAL = args.interval

    uvicorn.run(app, host="0.0.0.0", port=args.port)
