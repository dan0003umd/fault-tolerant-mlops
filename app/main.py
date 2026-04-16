from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import joblib
import time

app = FastAPI(title="Fault-Tolerant ML Inference Service")

model = joblib.load("model/model.joblib")

REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total prediction requests",
    ["method", "endpoint", "status"]
)

PREDICTION_ERRORS = Counter(
    "app_prediction_errors_total",
    "Total prediction errors",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Prediction latency in seconds",
    ["method", "endpoint", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

class IrisInput(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(input_data: IrisInput):
    start = time.time()
    method = "POST"
    endpoint = "/predict"
    status = "200"

    try:
        if len(input_data.features) != 4:
            status = "400"
            PREDICTION_ERRORS.labels(method=method, endpoint=endpoint, status=status).inc()
            raise HTTPException(status_code=400, detail="Expected exactly 4 features")

        prediction = model.predict([input_data.features]).tolist()
        latency = time.time() - start
        return {"prediction": prediction, "latency_seconds": latency}

    except HTTPException:
        raise

    except Exception:
        status = "500"
        PREDICTION_ERRORS.labels(method=method, endpoint=endpoint, status=status).inc()
        raise HTTPException(status_code=500, detail="Prediction failed")

    finally:
        latency = time.time() - start
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status=status).observe(latency)