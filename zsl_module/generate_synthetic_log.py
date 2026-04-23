"""
Generate synthetic requests.log matching Dhanraj's pipeline output format.
Use this to develop/test the ZSL module locally without needing EC2 access.

Output format (one JSON object per line):
  {"timestamp": "...", "mode": "normal|error_spike|slow",
   "load_level": "normal|high|critical", "latency_sec": 0.0087, "http_status": 200}
"""

import json
import random
import argparse
from datetime import datetime, timedelta

random.seed(42)


def generate_normal(ts: datetime, load_level: str) -> dict:
    """Normal mode: ~7.5ms avg latency, no errors."""
    latency = max(0.002, random.gauss(0.0075, 0.003))
    return {
        "timestamp": ts.isoformat(),
        "mode": "normal",
        "load_level": load_level,
        "latency_sec": round(latency, 6),
        "http_status": 200,
    }


def generate_error_spike(ts: datetime, load_level: str) -> dict:
    """Error spike mode: 30% chance of HTTP 500, otherwise normal latency."""
    is_error = random.random() < 0.30
    latency = max(0.002, random.gauss(0.0062, 0.003))
    return {
        "timestamp": ts.isoformat(),
        "mode": "error_spike",
        "load_level": load_level,
        "latency_sec": round(latency, 6),
        "http_status": 500 if is_error else 200,
    }


def generate_slow(ts: datetime, load_level: str) -> dict:
    """Slow mode: ~1.5s injected delay."""
    latency = max(1.0, random.gauss(1.508, 0.015))
    return {
        "timestamp": ts.isoformat(),
        "mode": "slow",
        "load_level": load_level,
        "latency_sec": round(latency, 6),
        "http_status": 200,
    }


# --- unseen fault types (NOT in Dhanraj's pipeline — for ZSL evaluation) ---

def generate_memory_leak(ts: datetime, load_level: str) -> dict:
    """Simulated memory leak: moderate latency (50-200ms range) with occasional 503s."""
    latency = max(0.03, random.gauss(0.12, 0.04))
    # 10% chance of 503 (service unavailable due to OOM)
    status = 503 if random.random() < 0.10 else 200
    return {
        "timestamp": ts.isoformat(),
        "mode": "memory_leak",
        "load_level": load_level,
        "latency_sec": round(latency, 6),
        "http_status": int(status),
    }


def generate_intermittent(ts: datetime, load_level: str) -> dict:
    """Intermittent fault: random mix of normal and spiky behavior."""
    if random.random() < 0.15:
        latency = max(0.5, random.gauss(0.8, 0.2))
        status = 500 if random.random() < 0.5 else 200
    else:
        latency = max(0.002, random.gauss(0.0075, 0.003))
        status = 200
    return {
        "timestamp": ts.isoformat(),
        "mode": "intermittent",
        "load_level": load_level,
        "latency_sec": round(latency, 6),
        "http_status": status,
    }


GENERATORS = {
    "normal": generate_normal,
    "error_spike": generate_error_spike,
    "slow": generate_slow,
    "memory_leak": generate_memory_leak,
    "intermittent": generate_intermittent,
}

# Experiment schedule matching Dhanraj's run_all_experiments.sh
EXPERIMENTS = [
    {"mode": "normal",      "load_level": "normal",   "rps": 2,  "duration_sec": 60},
    {"mode": "normal",      "load_level": "high",     "rps": 10, "duration_sec": 60},
    {"mode": "normal",      "load_level": "critical", "rps": 20, "duration_sec": 60},
    {"mode": "error_spike", "load_level": "normal",   "rps": 5,  "duration_sec": 60},
    {"mode": "slow",        "load_level": "normal",   "rps": 3,  "duration_sec": 60},
    # unseen faults for ZSL evaluation
    {"mode": "memory_leak",  "load_level": "normal",  "rps": 5,  "duration_sec": 60},
    {"mode": "intermittent", "load_level": "normal",  "rps": 5,  "duration_sec": 60},
]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic requests.log")
    parser.add_argument("--output", default="requests.log", help="Output file path")
    parser.add_argument("--known-only", action="store_true",
                        help="Only generate known fault types (normal/error_spike/slow)")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.known_only:
        experiments = [e for e in EXPERIMENTS if e["mode"] in ("normal", "error_spike", "slow")]

    records = []
    ts = datetime(2026, 4, 16, 17, 0, 0)

    for exp in experiments:
        gen_fn = GENERATORS[exp["mode"]]
        interval = 1.0 / exp["rps"]
        n_requests = int(exp["rps"] * exp["duration_sec"])

        # slow mode: fewer requests complete due to blocking
        if exp["mode"] == "slow":
            n_requests = int(exp["duration_sec"] / 1.5)

        for i in range(n_requests):
            record = gen_fn(ts, exp["load_level"])
            records.append(record)
            ts += timedelta(seconds=interval + random.uniform(-0.01, 0.01))

        # gap between experiments
        ts += timedelta(seconds=5)

    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # summary
    from collections import Counter
    mode_counts = Counter(r["mode"] for r in records)
    print(f"Generated {len(records)} records → {args.output}")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()