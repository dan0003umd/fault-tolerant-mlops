from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    from .actions import ActionResult
except ImportError:
    from actions import ActionResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Incident:
    incident_id: int
    fault_type: str
    load_level: str
    started_at: datetime
    last_seen_at: datetime
    recovery_started_at: datetime | None = None
    resolved_at: datetime | None = None
    mttr_seconds: float | None = None
    resolution_reason: str | None = None
    actions_attempted: int = 0
    successful_actions: int = 0
    degraded_samples: int = 0
    total_samples: int = 0
    actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["last_seen_at"] = self.last_seen_at.isoformat()
        data["recovery_started_at"] = (
            self.recovery_started_at.isoformat() if self.recovery_started_at else None
        )
        data["resolved_at"] = self.resolved_at.isoformat() if self.resolved_at else None
        return data


class RecoveryTracker:
    def __init__(self) -> None:
        self._next_incident_id = 1
        self.active_incident: Incident | None = None
        self.resolved_incidents: list[Incident] = []

        self.total_observations = 0
        self.healthy_observations = 0
        self.total_health_checks = 0
        self.successful_health_checks = 0

    def record_observation(self, prediction: str, app_healthy: bool) -> None:
        self.total_observations += 1
        if prediction == "normal":
            self.healthy_observations += 1
        if self.active_incident:
            self.active_incident.total_samples += 1
            if prediction != "normal":
                self.active_incident.degraded_samples += 1

        self.total_health_checks += 1
        if app_healthy:
            self.successful_health_checks += 1

    def start_or_update_incident(
        self,
        fault_type: str,
        load_level: str,
        observed_at: datetime | None = None,
    ) -> tuple[Incident, bool]:
        ts = observed_at or _utc_now()
        if self.active_incident is None:
            incident = Incident(
                incident_id=self._next_incident_id,
                fault_type=fault_type,
                load_level=load_level,
                started_at=ts,
                last_seen_at=ts,
            )
            self._next_incident_id += 1
            self.active_incident = incident
            return incident, True

        self.active_incident.last_seen_at = ts
        self.active_incident.fault_type = fault_type
        self.active_incident.load_level = load_level
        return self.active_incident, False

    def record_action(self, result: ActionResult, observed_at: datetime | None = None) -> None:
        if not self.active_incident:
            return
        ts = observed_at or _utc_now()

        incident = self.active_incident
        incident.actions_attempted += 1
        if result.success:
            incident.successful_actions += 1
            if incident.recovery_started_at is None:
                incident.recovery_started_at = ts
        incident.actions.append(result.to_dict())

    def resolve_active_incident(
        self,
        reason: str = "Fault condition cleared",
        observed_at: datetime | None = None,
    ) -> Incident | None:
        if self.active_incident is None:
            return None

        ts = observed_at or _utc_now()
        incident = self.active_incident
        incident.resolved_at = ts
        incident.resolution_reason = reason
        if incident.recovery_started_at:
            incident.mttr_seconds = (ts - incident.recovery_started_at).total_seconds()
        else:
            incident.mttr_seconds = (ts - incident.started_at).total_seconds()

        self.resolved_incidents.append(incident)
        self.active_incident = None
        return incident

    def summary(self) -> dict[str, Any]:
        mttrs = [
            inc.mttr_seconds
            for inc in self.resolved_incidents
            if inc.mttr_seconds is not None and inc.mttr_seconds >= 0
        ]
        availability = (
            self.healthy_observations / self.total_observations
            if self.total_observations
            else 1.0
        )
        app_health_availability = (
            self.successful_health_checks / self.total_health_checks
            if self.total_health_checks
            else 1.0
        )
        return {
            "active_incident": self.active_incident.to_dict() if self.active_incident else None,
            "resolved_incident_count": len(self.resolved_incidents),
            "mttr_seconds_avg": statistics.mean(mttrs) if mttrs else None,
            "mttr_seconds_median": statistics.median(mttrs) if mttrs else None,
            "availability_ratio": availability,
            "availability_impact_ratio": 1.0 - availability,
            "app_health_availability_ratio": app_health_availability,
            "total_observations": self.total_observations,
            "healthy_observations": self.healthy_observations,
        }

    def incident_history(self, limit: int = 50) -> list[dict[str, Any]]:
        items = self.resolved_incidents[-limit:]
        return [inc.to_dict() for inc in reversed(items)]
