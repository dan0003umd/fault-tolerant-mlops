import csv
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zsl_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "mean_latency", "max_latency", "p95_latency", "std_latency",
    "error_rate", "request_rate", "http_500_count",
    "cpu_percent", "mem_usage_mb", "mem_percent",
    "io_read_bytes", "io_write_bytes", "net_rx_bytes", "net_tx_bytes",
]
KNOWN_CLASSES = ["normal", "error_spike", "slow"]


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=14, embed_dim=16, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        emb = self.encoder(x)
        return emb, self.classifier(emb)

    def embed(self, x):
        with torch.no_grad():
            return self.encoder(x).numpy()


def _compute_val_sim_stats(val_sims):
    arr = np.array(val_sims, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def train():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "features.csv"))
    df_known = df[df["fault_type"].isin(KNOWN_CLASSES)].copy()

    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    y = le.transform(df_known["fault_type"].values)
    X = df_known[FEATURE_COLS].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = MLPEncoder(input_dim=14, embed_dim=16, n_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, 151):
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += float(loss.item())
        scheduler.step()

        model.eval()
        v_loss = 0.0
        v_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                _, logits = model(xb)
                v_loss += float(criterion(logits, yb).item())
                v_correct += int((logits.argmax(1) == yb).sum().item())

        val_acc = v_correct / len(X_val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": t_loss / len(train_dl),
                "val_loss": v_loss / len(val_dl),
                "val_acc": val_acc,
            }
        )

        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d} | val_loss={v_loss / len(val_dl):.4f} | val_acc={val_acc:.3f}")

    with open(os.path.join(OUTPUT_DIR, "training_history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    model.eval()
    train_embeddings = model.embed(torch.FloatTensor(X_train))
    centroids = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_train == cls_idx
        centroids[cls_name] = train_embeddings[mask].mean(axis=0).tolist()

    # Threshold from validation-set cosine similarities (not training set).
    val_embeddings = model.embed(torch.FloatTensor(X_val))
    val_sims = []
    for i, emb in enumerate(val_embeddings):
        cls_name = le.classes_[y_val[i]]
        centroid = np.array(centroids[cls_name])
        sim = float(cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0][0])
        val_sims.append(sim)

    stats = _compute_val_sim_stats(val_sims)
    raw_threshold = stats["p5"]
    threshold = min(raw_threshold, 0.75)

    print(
        "Val sims distribution "
        f"min={stats['min']:.6f} p5={stats['p5']:.6f} p25={stats['p25']:.6f} "
        f"p50={stats['p50']:.6f} p75={stats['p75']:.6f} max={stats['max']:.6f}"
    )
    print(f"Threshold (5th percentile, capped at 0.75): {threshold:.6f}")

    centroids["__threshold__"] = float(threshold)
    with open(os.path.join(OUTPUT_DIR, "centroids.json"), "w") as f:
        json.dump(centroids, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "embedding_model.pt"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
    print("Saved: embedding_model.pt, scaler.joblib, label_encoder.joblib, centroids.json")


if __name__ == "__main__":
    train()
