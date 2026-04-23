"""
Step 4: Evaluation
------------------
Compare ZSL detection against:
  1. Ground truth labels (accuracy, precision, recall, F1)
  2. Simple threshold baseline (latency > 500ms = fault)

Reports: detection accuracy, false positive rate, detection latency,
unseen fault detection rate.

Usage:
  python evaluate.py --predictions zsl_output/predictions.csv --output-dir zsl_output/
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


KNOWN_CLASSES = ["normal", "error_spike", "slow"]


def threshold_baseline(df: pd.DataFrame) -> list[str]:
    """
    Simple threshold-based fault detection:
      - error_rate > 0.1 → error_spike
      - mean_latency > 0.5s → slow
      - else → normal

    Cannot detect unseen faults by definition.
    """
    predictions = []
    for _, row in df.iterrows():
        if row["error_rate"] > 0.1:
            predictions.append("error_spike")
        elif row["mean_latency"] > 0.5:
            predictions.append("slow")
        else:
            predictions.append("normal")
    return predictions


def compute_metrics(y_true: list[str], y_pred: list[str], method_name: str) -> dict:
    """Compute and print metrics for a classification method."""
    # for fair comparison, map labels
    all_labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {acc:.3f}")

    print(f"\n  Classification Report:")
    report = classification_report(y_true, y_pred, labels=all_labels, zero_division=0)
    print(report)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print(f"  Confusion Matrix:")
    print(cm_df.to_string())

    # per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )

    # false positive rate for "fault detected" (anything non-normal)
    y_true_binary = [0 if y == "normal" else 1 for y in y_true]
    y_pred_binary = [0 if y == "normal" else 1 for y in y_pred]
    fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"\n  Fault Detection FPR: {fpr:.3f}")

    # unseen detection
    unseen_true = [y for y in y_true if y not in KNOWN_CLASSES]
    unseen_pred = [p for t, p in zip(y_true, y_pred) if t not in KNOWN_CLASSES]

    unseen_detection_rate = None
    if unseen_true:
        unseen_detected = sum(1 for p in unseen_pred if p == "UNSEEN")
        unseen_detection_rate = unseen_detected / len(unseen_true)
        print(f"  Unseen Fault Detection Rate: {unseen_detection_rate:.3f} "
              f"({unseen_detected}/{len(unseen_true)})")

    return {
        "method": method_name,
        "accuracy": acc,
        "fault_fpr": fpr,
        "unseen_detection_rate": unseen_detection_rate,
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(all_labels, precision, recall, f1, support)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ZSL vs baseline")
    parser.add_argument("--predictions", default="zsl_output/predictions.csv")
    parser.add_argument("--output-dir", default="zsl_output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    df = pd.read_csv(args.predictions)

    y_true = df["label"].tolist()
    y_zsl = df["zsl_prediction"].tolist()
    y_baseline = threshold_baseline(df)

    # Evaluate both methods
    zsl_metrics = compute_metrics(y_true, y_zsl, "Zero-Shot Learning (MLP Embeddings)")
    baseline_metrics = compute_metrics(y_true, y_baseline, "Threshold Baseline")

    # Comparative summary
    print("\n" + "=" * 60)
    print("  COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<30s} {'ZSL':>10s} {'Baseline':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':<30s} {zsl_metrics['accuracy']:>10.3f} {baseline_metrics['accuracy']:>10.3f}")
    print(f"  {'Fault Detection FPR':<30s} {zsl_metrics['fault_fpr']:>10.3f} {baseline_metrics['fault_fpr']:>10.3f}")

    if zsl_metrics["unseen_detection_rate"] is not None:
        zsl_udr = zsl_metrics["unseen_detection_rate"]
        base_udr = baseline_metrics["unseen_detection_rate"] or 0.0
        print(f"  {'Unseen Detection Rate':<30s} {zsl_udr:>10.3f} {base_udr:>10.3f}")

    # save
    results = {"zsl": zsl_metrics, "baseline": baseline_metrics}
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_dir}/evaluation_results.json")


if __name__ == "__main__":
    main()