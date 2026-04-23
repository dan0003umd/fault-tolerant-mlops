from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

try:
    from .actions import ActionExecutor
    from .policy import PolicyEngine
    from .recovery_metrics import RecoveryTracker
except ImportError:
    from actions import ActionExecutor
    from policy import PolicyEngine
    from recovery_metrics import RecoveryTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self_healing_controller")

APP_URL = os.getenv("APP_URL", "http://localhost:8000").rstrip("/")
ZSL_URL = os.getenv("ZSL_URL", "http://localhost:8001").rstrip("/")
MANAGED_CONTAINER = os.getenv("MANAGED_CONTAINER", "ftmlops-app")
APP_TARGETS = [
    item.strip().rstrip("/")
    for item in os.getenv("APP_TARGETS", APP_URL).split(",")
    if item.strip()
]
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "10"))
ACTION_COOLDOWN_SEC = int(os.getenv("ACTION_COOLDOWN_SEC", "30"))
MAX_ACTIONS_PER_INCIDENT = int(os.getenv("MAX_ACTIONS_PER_INCIDENT", "3"))
AUTO_START_MONITOR = os.getenv("AUTO_START_MONITOR", "true").lower() == "true"

monitor_task: asyncio.Task | None = None
monitor_running = False
last_action_unix_ts: float | None = None
last_classification: dict[str, Any] | None = None

app = FastAPI(
    title="Self-Healing Controller",
    description="Recovery policy service integrated with ZSL fault diagnosis",
    version="1.0.0",
)

policy_engine = PolicyEngine()
action_executor = ActionExecutor(app_targets=APP_TARGETS, managed_container=MANAGED_CONTAINER)
recovery_tracker = RecoveryTracker()

INCIDENTS_TOTAL = Counter(
    "controller_incidents_total",
    "Total incidents opened by fault type",
    ["fault_type"],
)
RECOVERY_ACTIONS_TOTAL = Counter(
    "controller_recovery_actions_total",
    "Recovery actions attempted",
    ["action", "result"],
)
RESOLVED_INCIDENTS_TOTAL = Counter(
    "controller_resolved_incidents_total",
    "Total incidents resolved by controller",
)
ACTIVE_INCIDENT_GAUGE = Gauge(
    "controller_active_incident",
    "Whether a recovery incident is currently active (0 or 1)",
)
LOAD_SHED_PERCENT_GAUGE = Gauge(
    "controller_load_shed_percent",
    "Current load shedding percentage",
)
AVAILABILITY_RATIO_GAUGE = Gauge(
    "controller_availability_ratio",
    "Observed availability ratio from fault classifications",
)
APP_HEALTH_AVAILABILITY_RATIO_GAUGE = Gauge(
    "controller_app_health_availability_ratio",
    "Observed availability ratio from app /health checks",
)
MTTR_SECONDS_HIST = Histogram(
    "controller_mttr_seconds",
    "Mean time to recovery in seconds for resolved incidents",
    buckets=[1, 2, 5, 10, 20, 30, 60, 120, 300, 600],
)


class IrisInput(BaseModel):
    features: list[float]


class PolicyUpdateRequest(BaseModel):
    policy_map: dict[str, dict[str, list[str]]] | None = None
    load_thresholds: dict[str, float] | None = None


class ManualRecoveryRequest(BaseModel):
    fault_type: str
    load_level: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)


class ManualLoadShedRequest(BaseModel):
    shed_percent: int = 30
    duration_sec: int = 45


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fetch_latest_classification() -> dict[str, Any] | None:
    global last_classification
    try:
        resp = requests.get(f"{ZSL_URL}/latest", timeout=4)
        if not resp.ok:
            return None
        payload = resp.json()
        if "prediction" not in payload:
            return None
        last_classification = payload
        return payload
    except Exception:
        return None


def _check_app_health() -> bool:
    try:
        resp = requests.get(f"{APP_URL}/health", timeout=3)
        return resp.ok
    except Exception:
        return False


def _update_gauges() -> dict[str, Any]:
    summary = recovery_tracker.summary()
    availability = float(summary["availability_ratio"])
    app_health_avail = float(summary["app_health_availability_ratio"])
    AVAILABILITY_RATIO_GAUGE.set(availability)
    APP_HEALTH_AVAILABILITY_RATIO_GAUGE.set(app_health_avail)

    action_state = action_executor.get_state()
    LOAD_SHED_PERCENT_GAUGE.set(action_state["load_shed_percent"])
    ACTIVE_INCIDENT_GAUGE.set(1 if recovery_tracker.active_incident else 0)
    return summary


def run_recovery_cycle(execute_actions: bool = True) -> dict[str, Any]:
    global last_action_unix_ts

    classification = _fetch_latest_classification()
    app_healthy = _check_app_health()

    prediction = "unknown"
    features: dict[str, Any] = {}
    if classification:
        prediction = str(classification.get("prediction", "unknown"))
        features = dict(classification.get("features", {}))

    recovery_tracker.record_observation(prediction=prediction, app_healthy=app_healthy)
    load_level = policy_engine.infer_load_level(features)

    opened_new_incident = False
    resolved_incident: dict[str, Any] | None = None
    action_results: list[dict[str, Any]] = []
    selected_actions: list[str] = []

    if prediction == "normal":
        resolved = recovery_tracker.resolve_active_incident(
            reason="ZSL returned normal state",
            observed_at=_utc_now(),
        )
        if resolved:
            RESOLVED_INCIDENTS_TOTAL.inc()
            if resolved.mttr_seconds is not None and resolved.mttr_seconds >= 0:
                MTTR_SECONDS_HIST.observe(resolved.mttr_seconds)
            resolved_incident = resolved.to_dict()
    elif prediction not in {"", "unknown"}:
        incident, opened_new_incident = recovery_tracker.start_or_update_incident(
            fault_type=prediction,
            load_level=load_level,
            observed_at=_utc_now(),
        )
        if opened_new_incident:
            INCIDENTS_TOTAL.labels(fault_type=prediction).inc()

        if execute_actions and incident.actions_attempted < MAX_ACTIONS_PER_INCIDENT:
            can_execute = (
                last_action_unix_ts is None
                or time.time() - last_action_unix_ts >= ACTION_COOLDOWN_SEC
            )
            if can_execute:
                selected_actions = policy_engine.select_actions(prediction, load_level)
                for action in selected_actions:
                    context = policy_engine.action_context(action, load_level, features)
                    result = action_executor.execute(action, context)
                    recovery_tracker.record_action(result, observed_at=_utc_now())
                    RECOVERY_ACTIONS_TOTAL.labels(
                        action=action,
                        result="success" if result.success else "failure",
                    ).inc()
                    action_results.append(result.to_dict())
                if selected_actions:
                    last_action_unix_ts = time.time()

    summary = _update_gauges()
    return {
        "timestamp": _utc_now().isoformat(),
        "prediction": prediction,
        "load_level": load_level,
        "app_healthy": app_healthy,
        "opened_new_incident": opened_new_incident,
        "resolved_incident": resolved_incident,
        "selected_actions": selected_actions,
        "action_results": action_results,
        "summary": summary,
    }


async def monitor_loop() -> None:
    logger.info(
        "Self-healing monitor started (interval=%ss, zsl=%s, app=%s)",
        POLL_INTERVAL_SEC,
        ZSL_URL,
        APP_URL,
    )
    while monitor_running:
        try:
            run_recovery_cycle(execute_actions=True)
        except Exception as exc:  # pragma: no cover
            logger.error("Monitor cycle failed: %s", exc)
        await asyncio.sleep(POLL_INTERVAL_SEC)


@app.on_event("startup")
async def startup_event() -> None:
    _update_gauges()
    if AUTO_START_MONITOR:
        await start_monitor_internal()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await stop_monitor_internal()


async def start_monitor_internal() -> dict[str, Any]:
    global monitor_task, monitor_running
    if monitor_running:
        return {"message": "monitor already running"}
    monitor_running = True
    monitor_task = asyncio.create_task(monitor_loop())
    return {"message": "monitor started", "interval_sec": POLL_INTERVAL_SEC}


async def stop_monitor_internal() -> dict[str, Any]:
    global monitor_task, monitor_running
    monitor_running = False
    if monitor_task and not monitor_task.done():
        monitor_task.cancel()
    monitor_task = None
    return {"message": "monitor stopped"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "status": "ok",
        "monitor_running": monitor_running,
        "config": {
            "app_url": APP_URL,
            "zsl_url": ZSL_URL,
            "poll_interval_sec": POLL_INTERVAL_SEC,
            "action_cooldown_sec": ACTION_COOLDOWN_SEC,
            "max_actions_per_incident": MAX_ACTIONS_PER_INCIDENT,
            "managed_container": MANAGED_CONTAINER,
            "app_targets": APP_TARGETS,
        },
        "last_classification": last_classification,
        "action_state": action_executor.get_state(),
        "summary": recovery_tracker.summary(),
    }


@app.get("/policy")
def get_policy() -> dict[str, Any]:
    return policy_engine.get_policy()


@app.put("/policy")
def update_policy(req: PolicyUpdateRequest) -> dict[str, Any]:
    return policy_engine.set_policy(req.policy_map, req.load_thresholds)


@app.post("/start_monitor")
async def start_monitor() -> dict[str, Any]:
    return await start_monitor_internal()


@app.post("/stop_monitor")
async def stop_monitor() -> dict[str, Any]:
    return await stop_monitor_internal()


@app.post("/trigger_once")
def trigger_once(execute_actions: bool = True) -> dict[str, Any]:
    return run_recovery_cycle(execute_actions=execute_actions)


@app.get("/summary")
def summary() -> dict[str, Any]:
    return recovery_tracker.summary()


@app.get("/incidents")
def incidents(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(500, limit))
    return {
        "active_incident": recovery_tracker.active_incident.to_dict()
        if recovery_tracker.active_incident
        else None,
        "resolved": recovery_tracker.incident_history(limit=limit),
    }


@app.post("/recover")
def recover(req: ManualRecoveryRequest) -> dict[str, Any]:
    fault_type = req.fault_type.strip()
    if not fault_type:
        raise HTTPException(status_code=400, detail="fault_type is required")

    load_level = req.load_level or policy_engine.infer_load_level(req.features)
    incident, opened = recovery_tracker.start_or_update_incident(
        fault_type=fault_type,
        load_level=load_level,
        observed_at=_utc_now(),
    )
    if opened:
        INCIDENTS_TOTAL.labels(fault_type=fault_type).inc()

    actions = policy_engine.select_actions(fault_type, load_level)
    results = []
    for action in actions:
        context = policy_engine.action_context(action, load_level, req.features)
        result = action_executor.execute(action, context)
        recovery_tracker.record_action(result, observed_at=_utc_now())
        RECOVERY_ACTIONS_TOTAL.labels(
            action=action, result="success" if result.success else "failure"
        ).inc()
        results.append(result.to_dict())

    _update_gauges()
    return {
        "incident_id": incident.incident_id,
        "fault_type": fault_type,
        "load_level": load_level,
        "actions_selected": actions,
        "results": results,
    }


@app.post("/load_shed")
def manual_load_shed(req: ManualLoadShedRequest) -> dict[str, Any]:
    result = action_executor.execute(
        "load_shed",
        {"shed_percent": req.shed_percent, "duration_sec": req.duration_sec},
    )
    RECOVERY_ACTIONS_TOTAL.labels(
        action="load_shed", result="success" if result.success else "failure"
    ).inc()
    _update_gauges()
    return result.to_dict()


@app.post("/reroute")
def manual_reroute() -> dict[str, Any]:
    result = action_executor.execute("traffic_reroute", {})
    RECOVERY_ACTIONS_TOTAL.labels(
        action="traffic_reroute", result="success" if result.success else "failure"
    ).inc()
    _update_gauges()
    return result.to_dict()


@app.post("/predict")
def proxy_predict(payload: IrisInput) -> dict[str, Any]:
    try:
        response = action_executor.proxy_predict(payload.model_dump(), timeout=5.0)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Proxy request failed: {exc}")

    if not response.ok:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Upstream app returned {response.status_code}",
        )

    body = response.json()
    body["proxied_target"] = action_executor.get_active_target()
    return body


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
