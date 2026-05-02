import time, random, logging
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import numpy as np

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="ML App — Iris Classifier")

# Train model at startup
iris = load_iris()
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(iris.data, iris.target)

# Prometheus metrics
REQUEST_COUNT   = Counter("app_requests_total","Total requests",["method","endpoint","status"])
ERROR_COUNT     = Counter("app_prediction_errors_total","Total prediction errors")
REQUEST_LATENCY = Histogram("app_request_latency_seconds","Request latency",["endpoint"],
                            buckets=[.005,.01,.025,.05,.1,.25,.5,1,2.5,5])

_mode = {"current": "normal"}   # normal | slow | error_spike | memory_leak

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    dur = time.time() - start
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(dur)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path,
                          status=response.status_code).inc()
    return response

@app.post("/predict")
def predict(data: dict):
    mode = _mode["current"]
    if mode == "slow":
        time.sleep(random.gauss(1.508, 0.015))
    elif mode == "memory_leak":
        time.sleep(max(0.03, random.gauss(0.12, 0.04)))
    from fastapi import HTTPException
    if mode == "error_spike" and random.random() < 0.30: 
        ERROR_COUNT.inc() 
        raise HTTPException(status_code=500, detail="prediction failed")
    features = data.get("features", [5.1, 3.5, 1.4, 0.2])
    pred = clf.predict([features])[0]
    return {"prediction": int(pred), "class": iris.target_names[pred], "mode": mode}

@app.get("/health")
def health():
    return {"status": "healthy", "mode": _mode["current"]}

@app.get("/getmode")
def getmode():
    return {"mode": _mode["current"]}

@app.post("/setmode/{mode}")
def setmode(mode: str):
    valid = ["normal","slow","error_spike","memory_leak"]
    if mode not in valid:
        return {"error": f"Invalid mode. Choose from {valid}"}, 400
    _mode["current"] = mode
    return {"mode": mode, "status": "set"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)