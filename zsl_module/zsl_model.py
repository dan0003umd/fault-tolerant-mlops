"""
Step 2 & 3: Embedding Model + Zero-Shot Inference
---------------------------------------------------
Uses scikit-learn MLPClassifier by default.
Pass --backend torch to use PyTorch (requires working torch install).

Usage:
  python zsl_model.py --features features.csv --output-dir zsl_output/
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

FEATURE_COLS = [
    "mean_latency", "max_latency", "p95_latency", "std_latency",
    "error_rate", "request_rate", "http_500_count",
]

KNOWN_CLASSES = ["normal", "error_spike", "slow"]


# =============================================
# Sklearn Embedding Model
# =============================================

class SklearnEmbeddingModel:
    """Wraps MLPClassifier; exposes get_embedding() via hidden activations."""

    def __init__(self, embed_dim=16):
        self.embed_dim = embed_dim
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32, embed_dim),
            activation="relu", max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, learning_rate_init=1e-3,
        )
        self._fitted = False

    def fit(self, X, y):
        self.mlp.fit(X, y)
        self._fitted = True
        return self

    def get_embedding(self, X):
        """Forward pass through hidden layers, return last hidden activations."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        a = np.asarray(X, dtype=np.float64)
        for W, b in zip(self.mlp.coefs_[:-1], self.mlp.intercepts_[:-1]):
            a = a @ W + b
            a = np.maximum(a, 0)  # ReLU
        return a

    def predict(self, X):
        return self.mlp.predict(X)

    def score(self, X, y):
        return self.mlp.score(X, y)


# =============================================
# Training
# =============================================

def train_sklearn_model(X_train, y_train, X_val, y_val, embed_dim=16):
    model = SklearnEmbeddingModel(embed_dim=embed_dim)
    print("  Training sklearn MLP (hidden_layers=64→32→16)...")
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Val accuracy:   {val_acc:.3f}")
    n_iter = model.mlp.n_iter_
    history = []
    for i in range(n_iter):
        loss_i = float(model.mlp.loss_curve_[min(i, len(model.mlp.loss_curve_) - 1)])
        history.append({"epoch": i + 1, "train_loss": loss_i,
                        "train_acc": train_acc, "val_loss": 0, "val_acc": val_acc})
    return model, history


def train_torch_model(X_train, y_train, X_val, y_val, embed_dim=16, epochs=150):
    """Only called when --backend torch is passed."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class EmbeddingMLP(nn.Module):
        def __init__(self, input_dim, embed_dim, n_classes):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, embed_dim),
            )
            self.classifier = nn.Linear(embed_dim, n_classes)

        def get_embedding(self, x):
            return self.encoder(x)

        def forward(self, x):
            return self.classifier(self.encoder(x))

    n_classes = len(np.unique(y_train))
    model = EmbeddingMLP(X_train.shape[1], embed_dim, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vl = model(X_val_t)
            val_loss = criterion(vl, y_val_t).item()
            val_acc = (vl.argmax(1) == y_val_t).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / total,
                        "train_acc": correct / total, "val_loss": val_loss, "val_acc": val_acc})
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d} | train_acc={correct/total:.3f} val_acc={val_acc:.3f}")

    return model, history


# =============================================
# Embeddings & ZSL
# =============================================

def get_embeddings(model, X):
    """Get embeddings from either backend."""
    return model.get_embedding(X)


def compute_centroids(model, X, y, label_encoder):
    embeddings = get_embeddings(model, X)
    centroids = {}
    for idx in range(len(label_encoder.classes_)):
        name = label_encoder.classes_[idx]
        mask = y == idx
        if mask.sum() > 0:
            centroids[name] = embeddings[mask].mean(axis=0)
    return centroids


def zsl_classify(model, X, centroids, threshold=0.5):
    """Classify via cosine similarity to centroids. Below threshold → UNSEEN."""
    embeddings = get_embeddings(model, X)
    centroid_names = list(centroids.keys())
    centroid_matrix = np.stack([centroids[n] for n in centroid_names])
    predictions, confidences = [], []
    for emb in embeddings:
        sims = cosine_similarity(emb.reshape(1, -1), centroid_matrix)[0]
        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]
        predictions.append("UNSEEN" if max_sim < threshold else centroid_names[max_idx])
        confidences.append(float(max_sim))
    return predictions, confidences


def calibrate_threshold(model, X_known, y_known, centroids, label_encoder, percentile=5):
    """Set threshold = Nth percentile of correct-class similarities on training data."""
    embeddings = get_embeddings(model, X_known)
    centroid_names = list(centroids.keys())
    centroid_matrix = np.stack([centroids[n] for n in centroid_names])
    correct_sims = []
    for i, emb in enumerate(embeddings):
        true_class = label_encoder.classes_[y_known[i]]
        if true_class in centroid_names:
            idx = centroid_names.index(true_class)
            sim = cosine_similarity(emb.reshape(1, -1), centroid_matrix[idx].reshape(1, -1))[0, 0]
            correct_sims.append(sim)
    threshold = float(np.percentile(correct_sims, percentile))
    print(f"  Calibrated threshold: {threshold:.4f} (p{percentile} of known-class similarities)")
    return threshold


# =============================================
# Main
# =============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.csv")
    parser.add_argument("--output-dir", default="zsl_output")
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--threshold-percentile", type=int, default=5)
    parser.add_argument("--backend", choices=["torch", "sklearn"], default="sklearn")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading features")
    print("=" * 60)
    df = pd.read_csv(args.features)
    print(f"  Total windows: {len(df)}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")
    print(f"  Backend: {args.backend}")

    known_mask = df["label"].isin(KNOWN_CLASSES)
    df_known = df[known_mask].copy()
    df_unseen = df[~known_mask].copy()
    print(f"  Known-class windows: {len(df_known)}")
    print(f"  Unseen-class windows: {len(df_unseen)}")

    scaler = StandardScaler()
    X_known = scaler.fit_transform(df_known[FEATURE_COLS].values)
    label_enc = LabelEncoder()
    y_known = label_enc.fit_transform(df_known["label"].values)
    print(f"  Classes: {list(label_enc.classes_)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_known, y_known, test_size=0.2, random_state=42, stratify=y_known)

    print("\n" + "=" * 60)
    print("STEP 2: Training embedding model")
    print("=" * 60)

    if args.backend == "torch":
        model, history = train_torch_model(X_train, y_train, X_val, y_val,
                                           embed_dim=args.embed_dim, epochs=args.epochs)
        import torch
        torch.save(model.state_dict(), out_dir / "embedding_model.pt")
    else:
        model, history = train_sklearn_model(X_train, y_train, X_val, y_val,
                                             embed_dim=args.embed_dim)
        joblib.dump(model, out_dir / "embedding_model.joblib")

    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(label_enc, out_dir / "label_encoder.joblib")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print("\n" + "=" * 60)
    print("STEP 3: Computing centroids & calibrating threshold")
    print("=" * 60)
    centroids = compute_centroids(model, X_train, y_train, label_enc)
    for name, c in centroids.items():
        print(f"  {name}: centroid norm = {np.linalg.norm(c):.4f}")

    threshold = calibrate_threshold(model, X_train, y_train, centroids, label_enc,
                                    percentile=args.threshold_percentile)
    centroid_data = {name: c.tolist() for name, c in centroids.items()}
    centroid_data["__threshold__"] = threshold
    with open(out_dir / "centroids.json", "w") as f:
        json.dump(centroid_data, f, indent=2)

    print("\n" + "=" * 60)
    print("STEP 4: Zero-Shot Classification")
    print("=" * 60)
    X_all = scaler.transform(df[FEATURE_COLS].values)
    predictions, confidences = zsl_classify(model, X_all, centroids, threshold=threshold)
    df["zsl_prediction"] = predictions
    df["zsl_confidence"] = confidences
    df.to_csv(out_dir / "predictions.csv", index=False)

    print("\n  Classification results:")
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        pred_counts = subset["zsl_prediction"].value_counts().to_dict()
        print(f"    True={label:15s} → Predicted: {pred_counts}")

    known_acc = (df[known_mask]["zsl_prediction"] == df[known_mask]["label"]).mean()
    print(f"\n  Known-class accuracy: {known_acc:.3f}")
    if len(df_unseen) > 0:
        unseen_detected = (df[~known_mask]["zsl_prediction"] == "UNSEEN").mean()
        print(f"  Unseen detection rate: {unseen_detected:.3f}")
    print(f"\n  Results saved to {out_dir}/predictions.csv")


if __name__ == "__main__":
    main()