"""
Step 1: Feature Extraction — parse requests.log into windowed feature vectors.
Optionally merges container-level metrics (CPU, memory, I/O) if available.
"""

import json
import argparse
import numpy as np
import pandas as pd


API_FEATURE_COLS = [
    "mean_latency", "max_latency", "p95_latency", "std_latency",
    "error_rate", "request_rate", "http_500_count",
]

CONTAINER_FEATURE_COLS = [
    "cpu_percent", "mem_usage_mb", "mem_percent",
    "io_read_bytes", "io_write_bytes", "net_rx_bytes", "net_tx_bytes",
]


def load_json_lines(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def extract_api_features(df, t0, window_sec):
    df = df.copy()
    df["elapsed_sec"] = (df["timestamp"] - t0).dt.total_seconds()
    df["window_id"] = (df["elapsed_sec"] // window_sec).astype(int)

    features = []
    for wid, group in df.groupby("window_id"):
        if len(group) < 2:
            continue
        latencies = group["latency_sec"].values
        statuses = group["http_status"].values
        features.append({
            "window_id": wid,
            "window_start": t0 + pd.Timedelta(seconds=wid * window_sec),
            "n_requests": len(group),
            "mean_latency": np.mean(latencies),
            "max_latency": np.max(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "std_latency": np.std(latencies),
            "error_rate": np.mean(statuses != 200),
            "request_rate": len(group) / window_sec,
            "http_500_count": int(np.sum(statuses == 500)),
            "label": group["mode"].mode().iloc[0],
            "load_level": group["load_level"].mode().iloc[0],
        })
    return pd.DataFrame(features)


def extract_container_features(df, t0, window_sec):
    df = df.copy()
    df["elapsed_sec"] = (df["timestamp"] - t0).dt.total_seconds()
    df["window_id"] = (df["elapsed_sec"] // window_sec).astype(int)

    features = []
    for wid, group in df.groupby("window_id"):
        if len(group) < 1:
            continue
        features.append({
            "window_id": wid,
            "cpu_percent": group["cpu_percent"].mean(),
            "mem_usage_mb": group["mem_usage_mb"].mean(),
            "mem_percent": group["mem_percent"].mean(),
            "io_read_bytes": group["io_read_bytes"].mean(),
            "io_write_bytes": group["io_write_bytes"].mean(),
            "net_rx_bytes": group["net_rx_bytes"].mean(),
            "net_tx_bytes": group["net_tx_bytes"].mean(),
        })
    return pd.DataFrame(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="requests.log")
    parser.add_argument("--container-log", default=None,
                        help="Path to container_metrics.log (optional)")
    parser.add_argument("--output", default="features.csv")
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    # Load API logs
    api_df = load_json_lines(args.input)
    print(f"Loaded {len(api_df)} API records from {args.input}")
    print(f"  Modes: {api_df['mode'].value_counts().to_dict()}")

    t0 = api_df["timestamp"].min()
    feat_df = extract_api_features(api_df, t0, args.window)

    # Merge container metrics if provided
    if args.container_log:
        cont_df = load_json_lines(args.container_log)
        print(f"Loaded {len(cont_df)} container metric records from {args.container_log}")
        cont_feat = extract_container_features(cont_df, t0, args.window)
        feat_df = feat_df.merge(cont_feat, on="window_id", how="left")
        for col in CONTAINER_FEATURE_COLS:
            if col in feat_df.columns:
                feat_df[col] = feat_df[col].fillna(0)
            else:
                feat_df[col] = 0
        print("  Merged container-level features")

    print(f"\nExtracted {len(feat_df)} windows (window_size={args.window}s)")
    print(f"  Labels:\n{feat_df['label'].value_counts().to_string()}")

    feat_df.to_csv(args.output, index=False)
    print(f"\nSaved features to {args.output}")


if __name__ == "__main__":
    main()