"""
zsl_server.py — ZSL inference server (port 8001)
Runtime-aligned ETL: all queries use sum() to avoid multi-series label mismatch.
"""
import os, json, time, logging, threading
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests as http_requests
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
logger = logging.getLogger("zsl_server")

OUTPUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zsl_output")
PROMETHEUS_URL  = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
ETL_INTERVAL    = int(os.getenv("ETL_INTERVAL", "5"))

FEATURE_COLS = [
    "mean_latency","max_latency","p95_latency","std_latency",
    "error_rate","request_rate","http_500_count",
    "cpu_percent","mem_usage_mb","mem_percent",
    "io_read_bytes","io_write_bytes","net_rx_bytes","net_tx_bytes",
]
KNOWN_CLASSES = ["normal","error_spike","slow"]
ANOMALOUS     = {"error_spike","slow","UNSEEN"}

# ── Model ─────────────────────────────────────────────────────
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=14, embed_dim=16, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        with torch.no_grad():
            return self.encoder(x).numpy()

# ── Load artefacts ────────────────────────────────────────────
model = MLPEncoder(14, 16, 3)
model.load_state_dict(
    torch.load(os.path.join(OUTPUT_DIR, "embedding_model.pt"), map_location="cpu"))
model.eval()
scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.joblib"))
with open(os.path.join(OUTPUT_DIR, "centroids.json")) as f:
    centroids = json.load(f)
threshold = centroids["__threshold__"]
logger.info(f"Model loaded. Threshold = {threshold:.6f}")

# Log scaler stats for debugging
logger.info(f"[SCALER] mean_={scaler.mean_.tolist()}")
logger.info(f"[SCALER] scale_={scaler.scale_.tolist()}")

# ── Prometheus metrics ────────────────────────────────────────
zsl_classifications_total = Counter("zsl_classifications_total","ZSL classifications",["label"])
zsl_unseen_total           = Counter("zsl_unseen_total","Total UNSEEN detections")
zsl_confidence             = Gauge("zsl_last_confidence","Confidence of last classification")
zsl_unseen_streak          = Gauge("zsl_unseen_streak","Consecutive UNSEEN windows")
zsl_mttr_seconds           = Gauge("zsl_mttr_seconds","Last measured MTTR in seconds")
zsl_incidents_total        = Counter("zsl_incidents_total","Total closed incidents")
zsl_detect_latency         = Histogram("zsl_detection_latency_seconds","Latency of classify call",
                                        buckets=[0.001,0.005,0.01,0.05,0.1,0.5,1.0])

# ── Shared state ──────────────────────────────────────────────
_lock           = threading.Lock()
_latest_result  = {}
_unseen_streak  = 0
_fault_onset_ts = None
_incident_log   = []

# ── Core classify ─────────────────────────────────────────────
def _do_classify(features: list) -> dict:
    global _unseen_streak, _fault_onset_ts

    x = np.array(features, dtype=np.float32).reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x = np.clip(x, -1e6, 1e6)
    x_scaled = scaler.transform(x)
    emb = model.embed(torch.FloatTensor(x_scaled))[0]

    sims = {}
    for cls in KNOWN_CLASSES:
        c = np.array(centroids[cls])
        sims[cls] = float(cosine_similarity(emb.reshape(1,-1), c.reshape(1,-1))[0][0])
    best_cls = max(sims, key=sims.get)
    best_sim = sims[best_cls]
    label    = "UNSEEN" if best_sim < threshold else best_cls

    with _lock:
        is_anomalous = label in ANOMALOUS
        if is_anomalous:
            if _fault_onset_ts is None:
                _fault_onset_ts = time.time()
                logger.warning(f"[MTTR] Fault onset label={label}")
            if label == "UNSEEN":
                _unseen_streak += 1
                zsl_unseen_total.inc()
            else:
                _unseen_streak = 0
        else:
            if _fault_onset_ts is not None:
                mttr = round(time.time() - _fault_onset_ts, 1)
                logger.info(f"[MTTR] Recovery confirmed MTTR={mttr}s")
                zsl_mttr_seconds.set(mttr)
                zsl_incidents_total.inc()
                _incident_log.append({
                    "onset_ts":    _fault_onset_ts,
                    "recovery_ts": time.time(),
                    "mttr_s":      mttr,
                    "last_label":  _latest_result.get("label","unknown"),
                })
                _fault_onset_ts = None
            _unseen_streak = 0

        zsl_unseen_streak.set(_unseen_streak)
        zsl_classifications_total.labels(label=label).inc()
        zsl_confidence.set(best_sim)

        result = {
            "label":               label,
            "confidence":          round(best_sim, 4),
            "unseen_flag":         label == "UNSEEN",
            "unseen_confidence":   round(best_sim, 4) if label == "UNSEEN" else None,
            "closest_known":       best_cls,
            "unseen_streak":       _unseen_streak,
            "retrain_recommended": _unseen_streak >= 5,
            "similarities":        {k: round(v, 4) for k, v in sims.items()},
            "threshold":           round(threshold, 6),
            "ts":                  time.time(),
        }
        _latest_result.update(result)
    return result

# ── Prometheus query helper ───────────────────────────────────
def _prom_query(q: str, default: float = 0.0) -> float:
    try:
        r    = http_requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                                  params={"query": q}, timeout=5)
        data = r.json()["data"]["result"]
        if not data:
            return default
        # Sum all returned series to handle multi-label metrics
        return sum(float(s["value"][1]) for s in data if s["value"][1] not in ("NaN","Inf","+Inf","-Inf"))
    except Exception:
        return default

# ── ETL feeder ────────────────────────────────────────────────
def _etl_feeder():
    logger.info("[ETL] Feeder started -- waiting 15s for Prometheus to stabilise")
    time.sleep(15)
    while True:
        try:
            # ── Feature 0: mean latency ──────────────────────
            total_lat = _prom_query(
                'sum(rate(app_request_latency_seconds_sum{endpoint="/predict"}[30s]))',
                default=0.0)
            total_req = _prom_query(
                'sum(rate(app_request_latency_seconds_count{endpoint="/predict"}[30s]))',
                default=0.0)
            f0_mean_lat = total_lat / (total_req + 0.001)

            # ── Feature 1: max latency (p99 proxy) ───────────
            f1_max_lat = _prom_query(
                'histogram_quantile(0.99,'
                ' sum(rate(app_request_latency_seconds_bucket{endpoint="/predict"}[30s]))'
                ' by (le))',
                default=0.01)

            # ── Feature 2: p95 latency ────────────────────────
            f2_p95_lat = _prom_query(
                'histogram_quantile(0.95,'
                ' sum(rate(app_request_latency_seconds_bucket{endpoint="/predict"}[30s]))'
                ' by (le))',
                default=0.01)

            # ── Feature 3: std latency (approx via var formula)─
            f3_std_lat = _prom_query(
                'stddev_over_time('
                'rate(app_request_latency_seconds_sum[5s])[30s:5s])',
                default=0.002)

            # ── Feature 4: error_rate ─────────────────────────
            # KEY FIX: sum() both sides so label sets match
            err_count = _prom_query(
                'sum(increase(app_prediction_errors_total[30s]))',
                default=0.0)
            req_count = _prom_query(
                'sum(increase(app_requests_total{endpoint="/predict"}[30s]))',
                default=1.0)
            f4_error_rate = err_count / (req_count + 0.001)

            # ── Feature 5: request_rate ───────────────────────
            f5_req_rate = _prom_query(
                'sum(rate(app_requests_total{endpoint="/predict"}[30s]))',
                default=0.0)

            # ── Feature 6: http_500_count ─────────────────────
            f6_http500 = _prom_query(
                'sum(increase(app_prediction_errors_total[30s]))',
                default=0.0)

            # ── Features 7-13: container metrics ─────────────
            f7_cpu  = _prom_query(
                'sum(rate(container_cpu_usage_seconds_total{name="ftmlops-app"}[30s])) * 100',
                default=5.0)
            f8_mem  = _prom_query(
                'sum(container_memory_usage_bytes{name="ftmlops-app"}) / 1048576',
                default=118.0)
            f9_mempct = _prom_query(
                'sum(container_memory_usage_bytes{name="ftmlops-app"})'
                ' / sum(container_spec_memory_limit_bytes{name="ftmlops-app"}) * 100',
                default=11.8)
            f10_ior = _prom_query(
                'sum(rate(container_fs_reads_bytes_total{name="ftmlops-app"}[30s]))',
                default=0.0)
            f11_iow = _prom_query(
                'sum(rate(container_fs_writes_bytes_total{name="ftmlops-app"}[30s]))',
                default=200.0)
            f12_netrx = _prom_query(
                'sum(rate(container_network_receive_bytes_total{name="ftmlops-app"}[30s]))',
                default=5000.0)
            f13_nettx = _prom_query(
                'sum(rate(container_network_transmit_bytes_total{name="ftmlops-app"}[30s]))',
                default=4000.0)

            window = [f0_mean_lat, f1_max_lat, f2_p95_lat, f3_std_lat,
                      f4_error_rate, f5_req_rate, f6_http500,
                      f7_cpu, f8_mem, f9_mempct,
                      f10_ior, f11_iow, f12_netrx, f13_nettx]

            # Always log raw features at INFO so we can see them
            logger.info(
                f"[ETL] raw_features err_rate={f4_error_rate:.4f} "
                f"req_rate={f5_req_rate:.2f} http500={f6_http500:.1f} "
                f"mean_lat={f0_mean_lat:.4f}"
            )

            t0     = time.time()
            result = _do_classify(window)
            zsl_detect_latency.observe(time.time() - t0)

            logger.info(
                f"[ETL] label={result['label']:<12} "
                f"conf={result['confidence']:.4f} "
                f"streak={result['unseen_streak']} "
                f"sims={result['similarities']}"
            )

        except Exception as e:
            logger.error(f"[ETL] Error: {e}", exc_info=True)

        time.sleep(ETL_INTERVAL)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="ZSL Fault Classifier", version="3.0")

class Window(BaseModel):
    mean_latency:  float = 0.0075
    max_latency:   float = 0.01
    p95_latency:   float = 0.01
    std_latency:   float = 0.002
    error_rate:    float = 0.0
    request_rate:  float = 10.0
    http_500_count:float = 0.0
    cpu_percent:   float = 5.0
    mem_usage_mb:  float = 118.0
    mem_percent:   float = 11.8
    io_read_bytes: float = 1000.0
    io_write_bytes:float = 500.0
    net_rx_bytes:  float = 10000.0
    net_tx_bytes:  float = 8000.0

@app.on_event("startup")
def startup():
    t = threading.Thread(target=_etl_feeder, daemon=True)
    t.start()
    logger.info("ETL feeder thread launched.")

@app.post("/classify")
def classify(w: Window):
    features = [getattr(w, col) for col in FEATURE_COLS]
    t0 = time.time()
    result = _do_classify(features)
    zsl_detect_latency.observe(time.time() - t0)
    return result

@app.get("/latest")
def latest():
    with _lock:
        return dict(_latest_result) if _latest_result else {
            "label":"normal","confidence":1.0,
            "unseen_flag":False,"unseen_streak":0,"retrain_recommended":False}

@app.get("/incidents")
def incidents():
    with _lock:
        log  = list(_incident_log)
        avg  = sum(i["mttr_s"] for i in log)/len(log) if log else 0.0
        return {"incidents": log, "total": len(log), "avg_mttr_s": round(avg,1)}

@app.get("/health")
def health():
    with _lock:
        return {
            "status":       "ok",
            "model_loaded": True,
            "threshold":    round(threshold, 6),
            "unseen_streak":_unseen_streak,
            "active_fault": _fault_onset_ts is not None,
        }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
