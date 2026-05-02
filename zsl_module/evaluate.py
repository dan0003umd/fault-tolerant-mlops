import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zsl_output")

FEATURE_COLS = [
    "mean_latency","max_latency","p95_latency","std_latency",
    "error_rate","request_rate","http_500_count",
    "cpu_percent","mem_usage_mb","mem_percent",
    "io_read_bytes","io_write_bytes","net_rx_bytes","net_tx_bytes"
]
KNOWN_CLASSES = ["normal","error_spike","slow"]
NOVEL_CLASSES = ["memory_leak","intermittent"]

class MLPEncoder(nn.Module):
    def __init__(self, input_dim=14, embed_dim=16, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        with torch.no_grad():
            return self.encoder(x).numpy()

def classify_window(emb, centroids, threshold, le):
    sims = {}
    for cls in KNOWN_CLASSES:
        c = np.array(centroids[cls])
        sims[cls] = float(cosine_similarity(emb.reshape(1,-1), c.reshape(1,-1))[0][0])
    best_cls = max(sims, key=sims.get)
    best_sim = sims[best_cls]
    if best_sim < threshold:
        return "UNSEEN", best_sim, best_cls, sims
    return best_cls, best_sim, best_cls, sims

def evaluate():
    df = pd.read_csv(os.path.join(OUTPUT_DIR,"features.csv"))

    scaler = joblib.load(os.path.join(OUTPUT_DIR,"scaler.joblib"))
    le     = joblib.load(os.path.join(OUTPUT_DIR,"label_encoder.joblib"))
    with open(os.path.join(OUTPUT_DIR,"centroids.json")) as f:
        centroids = json.load(f)
    threshold = centroids["__threshold__"]

    model = MLPEncoder(input_dim=14, embed_dim=16, n_classes=3)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR,"embedding_model.pt"),
                                      map_location="cpu"))
    model.eval()

    X = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    embeddings = model.embed(torch.FloatTensor(X))

    results = []
    for i, row in df.iterrows():
        emb = embeddings[i]
        ft = row["fault_type"]
        pred, conf, closest, sims = classify_window(emb, centroids, threshold, le)
        is_novel = ft in NOVEL_CLASSES
        correct = (pred == "UNSEEN") if is_novel else (pred == ft)
        results.append({
            "fault_type": ft, "predicted": pred, "confidence": round(conf,4),
            "closest_known": closest, "is_novel": is_novel, "correct": correct,
            **{f"sim_{k}": round(v,4) for k,v in sims.items()}
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR,"predictions.csv"), index=False)

    # ── Metrics ──────────────────────────────────────────────────────────────
    known_df = df_res[~df_res["is_novel"]]
    novel_df = df_res[df_res["is_novel"]]

    known_acc = known_df["correct"].mean()
    novel_acc = novel_df["correct"].mean()

    # False positive rate: known windows incorrectly flagged as UNSEEN
    fp = (known_df["predicted"] == "UNSEEN").sum()
    fpr = fp / len(known_df)

    # Per-class UNSEEN detection
    ml_df = df_res[df_res["fault_type"] == "memory_leak"]
    it_df = df_res[df_res["fault_type"] == "intermittent"]
    ml_detected = (ml_df["predicted"] == "UNSEEN").sum()
    it_detected = (it_df["predicted"] == "UNSEEN").sum()

    print("\n" + "="*60)
    print("  ZSL EVALUATION RESULTS")
    print("="*60)
    print(f"  Total windows evaluated : {len(df_res)} (52 known + 26 novel)")
    print(f"  Threshold               : {threshold:.10f}")
    print(f"")
    print(f"  Known-class accuracy    : {known_acc*100:.1f}%  ({known_df['correct'].sum()}/{len(known_df)})")
    print(f"  UNSEEN detection rate   : {novel_acc*100:.1f}%  ({novel_df['correct'].sum()}/{len(novel_df)})")
    print(f"    └─ memory_leak        : {ml_detected}/{len(ml_df)}")
    print(f"    └─ intermittent       : {it_detected}/{len(it_df)}")
    print(f"  False positive rate     : {fpr*100:.1f}%  ({fp}/{len(known_df)} known→UNSEEN)")
    print(f"")
    print(f"  Per known-class accuracy:")
    for cls in KNOWN_CLASSES:
        sub = known_df[known_df["fault_type"]==cls]
        acc = sub["correct"].mean() if len(sub) else 0
        print(f"    {cls:<15}: {acc*100:.1f}%  ({sub['correct'].sum()}/{len(sub)})")
    print("="*60)

    eval_results = {
        "total_windows": len(df_res),
        "known_windows": len(known_df),
        "novel_windows": len(novel_df),
        "threshold": threshold,
        "known_class_accuracy": round(known_acc, 4),
        "unseen_detection_rate": round(novel_acc, 4),
        "unseen_memory_leak": f"{ml_detected}/{len(ml_df)}",
        "unseen_intermittent": f"{it_detected}/{len(it_df)}",
        "false_positive_rate": round(fpr, 4),
        "false_positives": int(fp)
    }
    with open(os.path.join(OUTPUT_DIR,"evaluation_results.json"),"w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  Saved predictions.csv and evaluation_results.json")

if __name__ == "__main__":
    evaluate()