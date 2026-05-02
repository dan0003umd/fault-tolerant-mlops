import logging
import os
import threading
import time

import requests
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from starlette.responses import Response

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger("controller")

ZSL_URL = os.getenv("ZSL_URL", "http://zsl:8001").rstrip("/")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "5"))

app = FastAPI(title="Controller Metrics Exporter", version="1.0")

controller_last_mttr_seconds = Gauge(
    "controller_last_mttr_seconds",
    "Latest closed-incident MTTR in seconds",
)
controller_availability = Gauge(
    "controller_availability",
    "Service availability ratio in [0, 1]",
)
controller_active_incident = Gauge(
    "controller_active_incident",
    "1 when ZSL reports an active fault, else 0",
)
controller_incidents_closed_total = Gauge(
    "controller_incidents_closed_total",
    "Total number of closed incidents",
)

# Required startup state: initialize all gauges to 0.0
controller_last_mttr_seconds.set(0.0)
controller_availability.set(0.0)
controller_active_incident.set(0.0)
controller_incidents_closed_total.set(0.0)


def _safe_get_json(url: str):
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()


def _compute_availability(incidents):
    if not incidents:
        total_time = 3600.0
        total_downtime = 0.0
        return 1.0, total_time, total_downtime

    now = time.time()
    onset_times = [float(i.get("onset_ts", now)) for i in incidents]
    first_incident_ts = min(onset_times) if onset_times else now
    total_time = max(now - first_incident_ts, 1.0)
    total_downtime = float(sum(float(i.get("mttr_s", 0.0)) for i in incidents))
    availability = (total_time - total_downtime) / total_time
    availability = max(0.0, min(1.0, availability))
    return availability, total_time, total_downtime


def _poll_loop():
    while True:
        try:
            incidents_payload = _safe_get_json(f"{ZSL_URL}/incidents")
            health_payload = _safe_get_json(f"{ZSL_URL}/health")

            incidents = incidents_payload.get("incidents", []) or []
            closed_total = float(incidents_payload.get("total", len(incidents)))

            last_mttr = 0.0
            if incidents:
                last_mttr = float(incidents[-1].get("mttr_s", 0.0))

            availability, total_time, total_downtime = _compute_availability(incidents)
            active_fault = bool(health_payload.get("active_fault", False))
            active_incident = 1.0 if active_fault else 0.0

            controller_last_mttr_seconds.set(last_mttr)
            controller_availability.set(availability)
            controller_active_incident.set(active_incident)
            controller_incidents_closed_total.set(closed_total)

            logger.debug(
                "metrics update last_mttr=%.4f availability=%.6f active_incident=%.1f closed_total=%.1f total_time=%.2f total_downtime=%.2f",
                last_mttr,
                availability,
                active_incident,
                closed_total,
                total_time,
                total_downtime,
            )
        except Exception as e:
            logger.warning("poll failed: %s", e)

        time.sleep(POLL_INTERVAL)


@app.on_event("startup")
def startup():
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
