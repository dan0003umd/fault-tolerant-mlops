from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


@dataclass
class ActionResult:
    action: str
    success: bool
    message: str
    details: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionExecutor:
    def __init__(
        self,
        app_targets: list[str],
        managed_container: str = "ftmlops-app",
    ) -> None:
        self.app_targets = [target.rstrip("/") for target in app_targets if target.strip()]
        if not self.app_targets:
            self.app_targets = ["http://localhost:8000"]

        self.active_target_idx = 0
        self.managed_container = managed_container
        self.load_shed_percent = 0
        self.load_shed_until: datetime | None = None

    def get_active_target(self) -> str:
        self._refresh_load_shed()
        return self.app_targets[self.active_target_idx]

    def get_state(self) -> dict[str, Any]:
        self._refresh_load_shed()
        return {
            "managed_container": self.managed_container,
            "active_target": self.get_active_target(),
            "targets": self.app_targets,
            "load_shed_percent": self.load_shed_percent,
            "load_shed_until": self.load_shed_until.isoformat() if self.load_shed_until else None,
        }

    def should_shed_request(self) -> bool:
        self._refresh_load_shed()
        if self.load_shed_percent <= 0:
            return False
        return random.random() < (self.load_shed_percent / 100.0)

    def execute(self, action: str, context: dict[str, Any]) -> ActionResult:
        if action == "restart_container":
            return self._restart_container(action)
        if action == "traffic_reroute":
            return self._traffic_reroute(action)
        if action == "load_shed":
            return self._load_shed(action, context)

        return ActionResult(
            action=action,
            success=False,
            message=f"Unknown action: {action}",
            details={"context": context},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def proxy_predict(self, payload: dict[str, Any], timeout: float = 5.0) -> requests.Response:
        if self.should_shed_request():
            raise RuntimeError("Request dropped due to active load shedding policy.")
        target = self.get_active_target()
        return requests.post(
            f"{target}/predict",
            json=payload,
            timeout=timeout,
        )

    def _restart_container(self, action: str) -> ActionResult:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            import docker

            client = docker.from_env()
            container = client.containers.get(self.managed_container)
            container.restart(timeout=10)
            return ActionResult(
                action=action,
                success=True,
                message=f"Restarted container '{self.managed_container}'.",
                details={"container": self.managed_container},
                timestamp=ts,
            )
        except Exception as exc:  # pragma: no cover
            return ActionResult(
                action=action,
                success=False,
                message="Container restart failed.",
                details={"container": self.managed_container, "error": str(exc)},
                timestamp=ts,
            )

    def _traffic_reroute(self, action: str) -> ActionResult:
        ts = datetime.now(timezone.utc).isoformat()
        old_target = self.get_active_target()
        if len(self.app_targets) < 2:
            return ActionResult(
                action=action,
                success=False,
                message="Traffic reroute requires at least two targets.",
                details={"targets": self.app_targets},
                timestamp=ts,
            )

        self.active_target_idx = (self.active_target_idx + 1) % len(self.app_targets)
        new_target = self.get_active_target()
        return ActionResult(
            action=action,
            success=True,
            message="Traffic rerouted.",
            details={"from": old_target, "to": new_target},
            timestamp=ts,
        )

    def _load_shed(self, action: str, context: dict[str, Any]) -> ActionResult:
        ts = datetime.now(timezone.utc).isoformat()
        shed_percent = int(context.get("shed_percent", 30))
        duration_sec = int(context.get("duration_sec", 45))

        shed_percent = max(0, min(95, shed_percent))
        duration_sec = max(5, duration_sec)

        self.load_shed_percent = shed_percent
        self.load_shed_until = datetime.now(timezone.utc) + timedelta(seconds=duration_sec)

        return ActionResult(
            action=action,
            success=True,
            message="Load shedding activated.",
            details={
                "shed_percent": shed_percent,
                "duration_sec": duration_sec,
                "until": self.load_shed_until.isoformat(),
            },
            timestamp=ts,
        )

    def _refresh_load_shed(self) -> None:
        if self.load_shed_until and datetime.now(timezone.utc) >= self.load_shed_until:
            self.load_shed_percent = 0
            self.load_shed_until = None
