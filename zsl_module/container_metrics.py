"""
Container-level metrics collector.
Polls Docker API for CPU, memory, I/O, network stats from ftmlops-app container.
Writes metrics to container_metrics.log as JSON lines.
"""

import json
import time
import argparse
from datetime import datetime

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False


def collect_container_stats(container, duration_sec=60, interval_sec=2):
    """Poll container stats and return list of metric snapshots."""
    records = []
    start = time.time()
    print(f"  Collecting metrics for {duration_sec}s (every {interval_sec}s)...")

    while time.time() - start < duration_sec:
        stats = container.stats(stream=False)

        # CPU
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                    stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                       stats["precpu_stats"]["system_cpu_usage"]
        n_cpus = stats["cpu_stats"].get("online_cpus", 1)
        cpu_percent = (cpu_delta / system_delta) * n_cpus * 100.0 if system_delta > 0 else 0.0

        # Memory
        mem_usage = stats["memory_stats"].get("usage", 0)
        mem_limit = stats["memory_stats"].get("limit", 1)
        mem_percent = (mem_usage / mem_limit) * 100.0

        # Block I/O
        blkio = stats.get("blkio_stats", {}).get("io_service_bytes_recursive", []) or []
        io_read = sum(e["value"] for e in blkio if e["op"] == "read")
        io_write = sum(e["value"] for e in blkio if e["op"] == "write")

        # Network
        networks = stats.get("networks", {})
        net_rx = sum(v.get("rx_bytes", 0) for v in networks.values())
        net_tx = sum(v.get("tx_bytes", 0) for v in networks.values())

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": round(cpu_percent, 4),
            "mem_usage_mb": round(mem_usage / (1024 * 1024), 2),
            "mem_percent": round(mem_percent, 4),
            "io_read_bytes": io_read,
            "io_write_bytes": io_write,
            "net_rx_bytes": net_rx,
            "net_tx_bytes": net_tx,
        }
        records.append(record)
        elapsed = int(time.time() - start)
        print(f"    [{elapsed:3d}s] cpu={cpu_percent:.1f}% mem={record['mem_usage_mb']:.1f}MB", end="\r")
        time.sleep(interval_sec)

    print(f"\n  Collected {len(records)} snapshots")
    return records


def generate_synthetic_container_metrics(duration_sec=60, interval_sec=2, mode="normal"):
    """Generate synthetic container metrics for local testing without Docker."""
    import random
    random.seed(42)
    records = []
    n_snapshots = duration_sec // interval_sec

    for i in range(n_snapshots):
        ts = datetime(2026, 4, 16, 17, 0, 0)
        from datetime import timedelta
        ts = ts + timedelta(seconds=i * interval_sec)

        if mode == "normal":
            cpu = max(0, random.gauss(15, 5))
            mem = max(50, random.gauss(120, 10))
        elif mode == "high_load":
            cpu = max(0, random.gauss(65, 15))
            mem = max(50, random.gauss(200, 30))
        elif mode == "memory_leak":
            cpu = max(0, random.gauss(20, 5))
            mem = 120 + i * 3 + random.gauss(0, 5)  # steadily rising
        elif mode == "cpu_spike":
            cpu = max(0, random.gauss(85, 10))
            mem = max(50, random.gauss(130, 10))
        else:
            cpu = max(0, random.gauss(15, 5))
            mem = max(50, random.gauss(120, 10))

        records.append({
            "timestamp": ts.isoformat(),
            "cpu_percent": round(cpu, 4),
            "mem_usage_mb": round(mem, 2),
            "mem_percent": round(mem / 8192 * 100, 4),
            "io_read_bytes": int(random.gauss(50000, 10000)),
            "io_write_bytes": int(random.gauss(30000, 8000)),
            "net_rx_bytes": int(random.gauss(100000, 20000)),
            "net_tx_bytes": int(random.gauss(80000, 15000)),
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="ftmlops-app")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--output", default="container_metrics.log")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic metrics without Docker")
    parser.add_argument("--synthetic-mode", default="normal",
                        choices=["normal", "high_load", "memory_leak", "cpu_spike"])
    args = parser.parse_args()

    if args.synthetic:
        print(f"Generating synthetic container metrics (mode={args.synthetic_mode})...")
        records = generate_synthetic_container_metrics(
            args.duration, args.interval, args.synthetic_mode)
    else:
        if not HAS_DOCKER:
            print("ERROR: docker package not installed. Run: pip3 install docker")
            print("  Or use --synthetic flag for local testing.")
            return
        client = docker.from_env()
        container = client.containers.get(args.container)
        print(f"Collecting metrics from container '{args.container}'...")
        records = collect_container_stats(container, args.duration, args.interval)

    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()