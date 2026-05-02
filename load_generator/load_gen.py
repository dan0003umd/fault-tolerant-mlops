import argparse
import random
import statistics
import sys
import time

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--url",      default="http://localhost:8000")
parser.add_argument("--rps",      type=float, default=10)
parser.add_argument("--duration", type=int, default=60)
parser.add_argument("--mode",     default=None,
                    choices=["normal", "slow", "error_spike", "memory_leak"])
args = parser.parse_args()

if args.mode:
    try:
        requests.post(f"{args.url}/setmode/{args.mode}", timeout=3)
        print(f"Set fault mode: {args.mode}")
    except Exception as e:
        print(f"Could not set mode: {e}", file=sys.stderr)

interval = 1.0 / args.rps
count = 0
errors = 0
latencies = []

print(f"Sending continuous traffic at {args.rps} RPS to {args.url} (Ctrl+C to stop)")
print("-" * 50)

try:
    while True:
        tick = time.time()
        features = [
            round(random.uniform(4.0, 7.9), 1),
            round(random.uniform(2.0, 4.4), 1),
            round(random.uniform(1.0, 6.9), 1),
            round(random.uniform(0.1, 2.5), 1),
        ]
        try:
            t0 = time.time()
            r = requests.post(f"{args.url}/predict", json={"features": features}, timeout=10)
            latencies.append(time.time() - t0)
            print(f"sent={count + 1} status={r.status_code}")
            if r.status_code != 200:
                errors += 1
        except Exception as e:
            errors += 1
            print(f"sent={count + 1} status=EXCEPTION err={e}")
            latencies.append(10.0)

        count += 1
        elapsed = time.time() - tick
        time.sleep(max(0, interval - elapsed))
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

print("Done.")
print(f"  Requests  : {count}")
print(f"  Errors    : {errors}  ({(100 * errors / count) if count else 0:.1f}%)")
if latencies:
    print(f"  Mean lat  : {statistics.mean(latencies) * 1000:.1f} ms")
    p95_idx = max(0, int(0.95 * len(latencies)) - 1)
    print(f"  P95 lat   : {sorted(latencies)[p95_idx] * 1000:.1f} ms")

if args.mode:
    try:
        requests.post(f"{args.url}/setmode/normal", timeout=3)
        print("Reset mode to normal")
    except Exception:
        pass
