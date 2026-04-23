from __future__ import annotations

import copy
from typing import Any


DEFAULT_POLICY_MAP: dict[str, dict[str, list[str]]] = {
    "normal": {
        "normal": [],
        "high": [],
        "critical": [],
    },
    "error_spike": {
        "normal": ["restart_container"],
        "high": ["restart_container", "load_shed"],
        "critical": ["load_shed", "restart_container"],
    },
    "slow": {
        "normal": ["traffic_reroute"],
        "high": ["traffic_reroute", "load_shed"],
        "critical": ["load_shed", "traffic_reroute"],
    },
    "UNSEEN": {
        "normal": ["restart_container"],
        "high": ["restart_container", "traffic_reroute"],
        "critical": ["load_shed", "restart_container", "traffic_reroute"],
    },
}


DEFAULT_LOAD_THRESHOLDS = {
    "normal_max_rps": 5.0,
    "high_max_rps": 12.0,
}


class PolicyEngine:
    def __init__(
        self,
        policy_map: dict[str, dict[str, list[str]]] | None = None,
        load_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.policy_map = copy.deepcopy(policy_map or DEFAULT_POLICY_MAP)
        self.load_thresholds = dict(load_thresholds or DEFAULT_LOAD_THRESHOLDS)

    def get_policy(self) -> dict[str, Any]:
        return {
            "policy_map": copy.deepcopy(self.policy_map),
            "load_thresholds": dict(self.load_thresholds),
        }

    def set_policy(
        self,
        policy_map: dict[str, dict[str, list[str]]] | None,
        load_thresholds: dict[str, float] | None,
    ) -> dict[str, Any]:
        if policy_map is not None:
            self.policy_map = copy.deepcopy(policy_map)
        if load_thresholds is not None:
            self.load_thresholds = dict(load_thresholds)
        return self.get_policy()

    def infer_load_level(self, features: dict[str, Any]) -> str:
        explicit = str(features.get("load_level", "")).strip().lower()
        if explicit in {"normal", "high", "critical"}:
            return explicit

        request_rate = float(features.get("request_rate", 0.0))
        normal_max = float(self.load_thresholds.get("normal_max_rps", 5.0))
        high_max = float(self.load_thresholds.get("high_max_rps", 12.0))

        if request_rate <= normal_max:
            return "normal"
        if request_rate <= high_max:
            return "high"
        return "critical"

    def select_actions(self, fault_type: str, load_level: str) -> list[str]:
        fault = fault_type if fault_type in self.policy_map else "UNSEEN"
        level = load_level if load_level in {"normal", "high", "critical"} else "normal"
        return list(self.policy_map.get(fault, {}).get(level, []))

    def action_context(self, action: str, load_level: str, features: dict[str, Any]) -> dict[str, Any]:
        if action != "load_shed":
            return {"load_level": load_level, "features": features}

        if load_level == "critical":
            return {"shed_percent": 60, "duration_sec": 75}
        if load_level == "high":
            return {"shed_percent": 40, "duration_sec": 60}
        return {"shed_percent": 20, "duration_sec": 45}
