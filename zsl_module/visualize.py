"""
Visualization — embedding space, training curves, confusion matrices, confidence distributions.

Usage:
  python visualize.py --features features.csv --output-dir zsl_output/
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from zsl_model import FEATURE_COLS, get_embeddings, SklearnEmbeddingModel


def load_model(out_dir):
    """Load whichever model backend was saved."""
    sk_path = out_dir / "embedding_model.joblib"
    pt_path = out_dir / "embedding_model.pt"

    if sk_path.exists():
        return joblib.load(sk_path)
    elif pt_path.exists():
        import torch
        from zsl_model import train_torch_model  # triggers EmbeddingMLP def
        # reconstruct — need label_enc for n_classes
        label_enc = joblib.load(out_dir / "label_encoder.joblib")
        n_classes = len(label_enc.classes_)

        import torch.nn as nn
        class EmbeddingMLP(nn.Module):
            def __init__(self, input_dim, embed_dim, n_classes):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, embed_dim))
                self.classifier = nn.Linear(embed_dim, n_classes)
            def get_embedding(self, x): return self.encoder(x)
            def forward(self, x): return self.classifier(self.encoder(x))

        model = EmbeddingMLP(len(FEATURE_COLS), 16, n_classes)
        model.load_state_dict(torch.load(pt_path, weights_only=True))
        return model
    raise FileNotFoundError("No model found in " + str(out_dir))


def plot_training_curves(history_path, out_dir):
    hist = pd.read_csv(history_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist["epoch"], hist["train_loss"], label="Train Loss", lw=2)
    if hist["val_loss"].sum() > 0:
        ax1.plot(hist["epoch"], hist["val_loss"], label="Val Loss", lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(hist["epoch"], hist["train_acc"], label="Train Acc", lw=2)
    if hist["val_acc"].sum() > 0:
        ax2.plot(hist["epoch"], hist["val_acc"], label="Val Acc", lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Training Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved training_curves.png")


def plot_embedding_space(model, X, labels, out_dir):
    embeddings = get_embeddings(model, X)
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(embeddings)
    for i, label in enumerate(unique_labels):
        mask = np.array([l == label for l in labels])
        ax1.scatter(emb_pca[mask, 0], emb_pca[mask, 1], label=label, alpha=0.7, s=50,
                    edgecolors="white", linewidth=0.5, color=palette[i])
    ax1.set_title("Embedding Space (PCA)")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    if len(embeddings) > 5:
        perp = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        emb_tsne = tsne.fit_transform(embeddings)
        for i, label in enumerate(unique_labels):
            mask = np.array([l == label for l in labels])
            ax2.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1], label=label, alpha=0.7, s=50,
                        edgecolors="white", linewidth=0.5, color=palette[i])
    ax2.set_title("Embedding Space (t-SNE)"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_space.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved embedding_space.png")


def plot_confusion_matrices(predictions_path, out_dir):
    df = pd.read_csv(predictions_path)
    baseline_preds = []
    for _, row in df.iterrows():
        if row["error_rate"] > 0.1: baseline_preds.append("error_spike")
        elif row["mean_latency"] > 0.5: baseline_preds.append("slow")
        else: baseline_preds.append("normal")
    df["baseline_prediction"] = baseline_preds

    all_labels = sorted(set(df["label"]) | set(df["zsl_prediction"]) | set(baseline_preds))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cm_zsl = confusion_matrix(df["label"], df["zsl_prediction"], labels=all_labels)
    sns.heatmap(cm_zsl, annot=True, fmt="d", cmap="Blues",
                xticklabels=all_labels, yticklabels=all_labels, ax=ax1)
    ax1.set_title("Zero-Shot Learning"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")

    cm_base = confusion_matrix(df["label"], df["baseline_prediction"], labels=all_labels)
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Oranges",
                xticklabels=all_labels, yticklabels=all_labels, ax=ax2)
    ax2.set_title("Threshold Baseline"); ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")

    plt.suptitle("Confusion Matrices: ZSL vs Threshold Baseline", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved confusion_matrices.png")


def plot_confidence_distribution(predictions_path, out_dir):
    df = pd.read_csv(predictions_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        ax.hist(subset["zsl_confidence"], bins=20, alpha=0.5, label=label, density=True)

    centroids_path = Path(predictions_path).parent / "centroids.json"
    if centroids_path.exists():
        with open(centroids_path) as f:
            cdata = json.load(f)
        threshold = cdata.get("__threshold__", 0.5)
        ax.axvline(x=threshold, color="red", linestyle="--", lw=2,
                    label=f"UNSEEN threshold ({threshold:.2f})")

    ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Density")
    ax.set_title("ZSL Confidence Distribution by True Class")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved confidence_distribution.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.csv")
    parser.add_argument("--output-dir", default="zsl_output")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    model = load_model(out_dir)
    scaler = joblib.load(out_dir / "scaler.joblib")
    df = pd.read_csv(args.features)
    X = scaler.transform(df[FEATURE_COLS].values)

    print("Generating visualizations...")
    plot_training_curves(str(out_dir / "training_history.csv"), out_dir)
    plot_embedding_space(model, X, df["label"].tolist(), out_dir)
    plot_confusion_matrices(str(out_dir / "predictions.csv"), out_dir)
    plot_confidence_distribution(str(out_dir / "predictions.csv"), out_dir)
    print("\nAll visualizations saved to", out_dir)


if __name__ == "__main__":
    main()