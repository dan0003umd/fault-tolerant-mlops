import argparse
import time
import random
import requests
from statistics import mean

def run_load(url: str, rps: float, duration: int, mode: str, load_level: str):
    interval = 1.0 / rps if rps > 0 else 0
    features = [5.1, 3.5, 1.4, 0.2]

    latencies = []
    errors = 0
    total = 0

    print(f"Starting load: url={url}, rps={rps}, duration={duration}s, mode={mode}, load_level={load_level}")

    end_time = time.time() + duration
    while time.time() < end_time:
        start = time.time()
        try:
            # Optional: include load_level as query param for future logging
            resp = requests.post(
                url,
                params={"load_level": load_level},
                json={"features": features},
                timeout=5,
            )
            latency = time.time() - start
            latencies.append(latency)
            total += 1

            if not resp.ok:
                errors += 1

        except Exception as e:
            errors += 1
            total += 1
            print(f"Request error: {e}")

        # sleep to maintain approximate RPS
        elapsed = time.time() - start
        to_sleep = interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    avg_latency = mean(latencies) if latencies else 0.0
    error_rate = errors / total if total else 0.0

    print("\n=== Load run summary ===")
    print(f"Total requests: {total}")
    print(f"Errors:         {errors}")
    print(f"Error rate:     {error_rate:.2%}")
    print(f"Avg latency:    {avg_latency:.4f} sec")
    if latencies:
        print(f"Min latency:    {min(latencies):.4f} sec")
        print(f"Max latency:    {max(latencies):.4f} sec")


def main():
    parser = argparse.ArgumentParser(description="Simple load generator for FastAPI ML service")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict",
                        help="Target /predict URL")
    parser.add_argument("--rps", type=float, default=5.0,
                        help="Requests per second")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of the test in seconds")
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "high", "critical"],
                        help="Logical load mode label")
    args = parser.parse_args()

    # Map logical mode → default RPS if user didn't override
    if "rps" not in vars(args) or args.rps is None:
        if args.mode == "normal":
            rps = 2.0
        elif args.mode == "high":
            rps = 10.0
        else:  # critical
            rps = 20.0
    else:
        rps = args.rps

    run_load(url=args.url, rps=rps, duration=args.duration, mode=args.mode, load_level=args.mode)


if __name__ == "__main__":
    main()