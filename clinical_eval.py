"""Comprehensive clinical evaluation for chest X-ray pneumonia detection.

Generates publication-quality plots and metrics that a clinician or
reviewer would expect to see:

- ROC curve with AUC and operating points
- Precision-Recall curve with Average Precision
- Confusion matrix heatmap (counts + percentages)
- Calibration curve / reliability diagram
- Operating-point analysis (screening vs. confirmation thresholds)
- Confidence distribution histograms per class
- Bootstrap confidence intervals for sensitivity, specificity, PPV, NPV

Usage::

    python clinical_eval.py \\
        --checkpoint artifacts/improved_resnet18.pt \\
        --data-dir data/chest-xray-pneumonia/chest_xray \\
        --device mps --num-workers 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from modeling import load_checkpoint, make_dataloaders, resolve_data_root, select_device

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

_COLORS = {
    "primary": "#4F8FF7",
    "secondary": "#F74F4F",
    "accent": "#34D399",
    "warning": "#FBBF24",
    "bg": "#F8FAFC",
    "grid": "#E2E8F0",
    "text": "#1E293B",
    "text_light": "#64748B",
}


def _apply_style() -> None:
    """Apply a clean, publication-quality plot style."""
    plt.rcParams.update({
        "figure.facecolor": _COLORS["bg"],
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": _COLORS["grid"],
        "axes.labelcolor": _COLORS["text"],
        "axes.grid": True,
        "grid.color": _COLORS["grid"],
        "grid.alpha": 0.6,
        "xtick.color": _COLORS["text_light"],
        "ytick.color": _COLORS["text_light"],
        "text.color": _COLORS["text"],
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": _COLORS["bg"],
    })


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    pneumonia_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on *loader* and return (binary_labels, pneumonia_probs)."""
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, pneumonia_index].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    # Convert to binary: 1 = pneumonia, 0 = normal
    binary_labels = (labels_arr == pneumonia_index).astype(int)
    return binary_labels, probs_arr


# ---------------------------------------------------------------------------
# Individual plot generators
# ---------------------------------------------------------------------------

def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    """ROC curve with AUC and key operating points."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color=_COLORS["primary"], lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=_COLORS["text_light"], lw=1, ls="--", alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=_COLORS["primary"])

    # Mark operating points
    ax.plot(fpr[best_idx], tpr[best_idx], "o", ms=10, color=_COLORS["secondary"],
            label=f"Optimal (t={best_threshold:.2f})", zorder=5)

    # Mark default 0.5 threshold
    idx_05 = int(np.argmin(np.abs(thresholds - 0.5)))
    ax.plot(fpr[idx_05], tpr[idx_05], "s", ms=9, color=_COLORS["accent"],
            label=f"Default (t=0.50)", zorder=5)

    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.savefig(output_path)
    plt.close(fig)

    return {
        "auc_roc": round(roc_auc, 4),
        "optimal_threshold": round(best_threshold, 4),
        "optimal_sensitivity": round(float(tpr[best_idx]), 4),
        "optimal_specificity": round(1 - float(fpr[best_idx]), 4),
    }


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    """Precision-Recall curve with Average Precision."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, color=_COLORS["primary"], lw=2.5, label=f"PR (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.08, color=_COLORS["primary"])

    # Prevalence line (baseline)
    prevalence = labels.mean()
    ax.axhline(y=prevalence, color=_COLORS["text_light"], lw=1, ls="--", alpha=0.5,
               label=f"Prevalence = {prevalence:.2f}")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.savefig(output_path)
    plt.close(fig)

    return {"average_precision": round(ap, 4), "prevalence": round(float(prevalence), 4)}


def plot_confusion_matrix(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    output_path: Path,
) -> dict[str, int]:
    """Confusion matrix heatmap with counts and percentages."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Heatmap
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Pneumonia"], fontsize=12)
    ax.set_yticklabels(["Normal", "Pneumonia"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100
            color = "white" if count > total * 0.3 else _COLORS["text"]
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path)
    plt.close(fig)

    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def plot_calibration_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> dict[str, float]:
    """Calibration / reliability diagram."""
    fraction_positives, mean_predicted = calibration_curve(labels, probs, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(len(fraction_positives)):
        bin_lower, bin_upper = bin_edges[i], bin_edges[i + 1]
        in_bin = ((probs >= bin_lower) & (probs < bin_upper))
        n_in_bin = in_bin.sum()
        if n_in_bin > 0:
            ece += (n_in_bin / len(probs)) * abs(fraction_positives[i] - mean_predicted[i])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: reliability diagram
    ax1.plot([0, 1], [0, 1], ls="--", color=_COLORS["text_light"], lw=1, label="Perfectly calibrated")
    ax1.plot(mean_predicted, fraction_positives, "o-", color=_COLORS["primary"], lw=2,
             ms=8, label=f"Model (ECE = {ece:.3f})")
    ax1.fill_between(mean_predicted, fraction_positives, mean_predicted, alpha=0.1,
                     color=_COLORS["secondary"])
    ax1.set_ylabel("Fraction of Positives (Actual)")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Bottom: histogram of predictions
    ax2.hist(probs, bins=n_bins, range=(0, 1), color=_COLORS["primary"], alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    fig.tight_layout(pad=2)
    fig.savefig(output_path)
    plt.close(fig)

    return {"ece": round(ece, 4)}


def plot_confidence_distribution(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
) -> None:
    """Histogram of predicted pneumonia probabilities split by true class."""
    fig, ax = plt.subplots(figsize=(8, 5))

    normal_probs = probs[labels == 0]
    pneumonia_probs = probs[labels == 1]

    bins = np.linspace(0, 1, 30)
    ax.hist(normal_probs, bins=bins, alpha=0.65, color=_COLORS["accent"],
            label=f"Normal (n={len(normal_probs)})", edgecolor="white")
    ax.hist(pneumonia_probs, bins=bins, alpha=0.65, color=_COLORS["secondary"],
            label=f"Pneumonia (n={len(pneumonia_probs)})", edgecolor="white")

    ax.axvline(x=0.5, color=_COLORS["text_light"], lw=1.5, ls="--", label="Decision threshold")

    ax.set_xlabel("Predicted Pneumonia Probability")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution by True Class")
    ax.legend(framealpha=0.9)

    fig.savefig(output_path)
    plt.close(fig)


def plot_operating_points(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    """Sensitivity/Specificity trade-off across thresholds with clinical operating points."""
    thresholds = np.linspace(0.01, 0.99, 200)
    sensitivities = []
    specificities = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)

    # Find clinical operating points
    # Screening: sensitivity >= 0.95, maximize specificity
    screening_mask = sensitivities >= 0.95
    if screening_mask.any():
        screening_idx = int(np.argmax(specificities[screening_mask]))
        screening_idx = np.where(screening_mask)[0][screening_idx]
    else:
        screening_idx = int(np.argmax(sensitivities))

    # Confirmation: specificity >= 0.95, maximize sensitivity
    confirm_mask = specificities >= 0.95
    if confirm_mask.any():
        confirm_idx = int(np.argmax(sensitivities[confirm_mask]))
        confirm_idx = np.where(confirm_mask)[0][confirm_idx]
    else:
        confirm_idx = int(np.argmax(specificities))

    # Balanced: maximize Youden's J
    j_scores = sensitivities + specificities - 1
    balanced_idx = int(np.argmax(j_scores))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(thresholds, sensitivities, color=_COLORS["secondary"], lw=2, label="Sensitivity")
    ax.plot(thresholds, specificities, color=_COLORS["primary"], lw=2, label="Specificity")

    # Mark operating points
    for idx, name, marker, color in [
        (screening_idx, "Screening", "^", _COLORS["accent"]),
        (balanced_idx, "Balanced", "o", _COLORS["warning"]),
        (confirm_idx, "Confirmation", "s", "#8B5CF6"),
    ]:
        ax.axvline(x=thresholds[idx], color=color, lw=1, ls=":", alpha=0.6)
        ax.plot(thresholds[idx], sensitivities[idx], marker, ms=11, color=color, zorder=5)
        ax.plot(thresholds[idx], specificities[idx], marker, ms=11, color=color, zorder=5)
        ax.annotate(
            f"{name}\nt={thresholds[idx]:.2f}",
            xy=(thresholds[idx], max(sensitivities[idx], specificities[idx])),
            xytext=(0, 15), textcoords="offset points",
            fontsize=9, ha="center", fontweight="bold", color=color,
        )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Operating Point Analysis: Sensitivity vs. Specificity")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)

    fig.savefig(output_path)
    plt.close(fig)

    return {
        "screening": {
            "threshold": round(float(thresholds[screening_idx]), 4),
            "sensitivity": round(float(sensitivities[screening_idx]), 4),
            "specificity": round(float(specificities[screening_idx]), 4),
        },
        "balanced": {
            "threshold": round(float(thresholds[balanced_idx]), 4),
            "sensitivity": round(float(sensitivities[balanced_idx]), 4),
            "specificity": round(float(specificities[balanced_idx]), 4),
        },
        "confirmation": {
            "threshold": round(float(thresholds[confirm_idx]), 4),
            "sensitivity": round(float(sensitivities[confirm_idx]), 4),
            "specificity": round(float(specificities[confirm_idx]), 4),
        },
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap 95% confidence intervals for clinical metrics."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    alpha = (1 - confidence) / 2

    metrics_boot: dict[str, list[float]] = {
        "sensitivity": [], "specificity": [], "ppv": [], "npv": [], "f1": [], "accuracy": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bl = labels[idx]
        bp = (probs[idx] >= threshold).astype(int)

        cm = confusion_matrix(bl, bp, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()

        metrics_boot["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        metrics_boot["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics_boot["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics_boot["npv"].append(tn / (tn + fn) if (tn + fn) > 0 else 0)

        _, _, f1, _ = precision_recall_fscore_support(bl, bp, average="binary", zero_division=0)
        metrics_boot["f1"].append(float(f1))
        metrics_boot["accuracy"].append((tp + tn) / (tp + tn + fp + fn))

    result: dict[str, dict[str, float]] = {}
    for metric, values in metrics_boot.items():
        arr = np.array(values)
        result[metric] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "ci_lower": round(float(np.percentile(arr, alpha * 100)), 4),
            "ci_upper": round(float(np.percentile(arr, (1 - alpha) * 100)), 4),
        }

    return result


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary_report(all_results: dict, output_path: Path) -> str:
    """Generate a human-readable summary of all evaluation results."""
    roc = all_results["roc"]
    pr = all_results["precision_recall"]
    cal = all_results["calibration"]
    cm = all_results["confusion_matrix"]
    ops = all_results["operating_points"]
    boot = all_results["bootstrap"]

    lines = [
        "=" * 65,
        "         CLINICAL EVALUATION REPORT",
        "         Pneumonia Detection Model",
        "=" * 65,
        "",
        "─── OVERALL PERFORMANCE ───────────────────────────────────────",
        f"  AUC-ROC:              {roc['auc_roc']:.4f}",
        f"  Average Precision:    {pr['average_precision']:.4f}",
        f"  Calibration ECE:      {cal['ece']:.4f}",
        f"  Disease Prevalence:   {pr['prevalence']:.1%}",
        "",
        "─── CONFUSION MATRIX (threshold=0.50) ────────────────────────",
        f"  True Positives:   {cm['tp']:>5d}    False Positives:  {cm['fp']:>5d}",
        f"  True Negatives:   {cm['tn']:>5d}    False Negatives:  {cm['fn']:>5d}",
        "",
        "─── METRICS WITH 95% CONFIDENCE INTERVALS ────────────────────",
    ]

    for metric in ["sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"]:
        b = boot[metric]
        lines.append(
            f"  {metric.upper():<16s} {b['mean']:.1%}  "
            f"(95% CI: {b['ci_lower']:.1%} – {b['ci_upper']:.1%})"
        )

    lines += [
        "",
        "─── OPERATING POINTS ──────────────────────────────────────────",
        f"  SCREENING    (high sensitivity):  t={ops['screening']['threshold']:.2f}  "
        f"sens={ops['screening']['sensitivity']:.1%}  spec={ops['screening']['specificity']:.1%}",
        f"  BALANCED     (Youden's J):        t={ops['balanced']['threshold']:.2f}  "
        f"sens={ops['balanced']['sensitivity']:.1%}  spec={ops['balanced']['specificity']:.1%}",
        f"  CONFIRMATION (high specificity):  t={ops['confirmation']['threshold']:.2f}  "
        f"sens={ops['confirmation']['sensitivity']:.1%}  spec={ops['confirmation']['specificity']:.1%}",
        "",
        f"  Optimal threshold (ROC): {roc['optimal_threshold']:.2f}  "
        f"sens={roc['optimal_sensitivity']:.1%}  spec={roc['optimal_specificity']:.1%}",
        "",
        "─── GENERATED PLOTS ───────────────────────────────────────────",
        "  • roc_curve.png              — ROC with AUC and operating points",
        "  • precision_recall_curve.png  — PR curve with Average Precision",
        "  • confusion_matrix.png        — Heatmap with counts and percentages",
        "  • calibration_curve.png       — Reliability diagram + distribution",
        "  • confidence_distribution.png — Probability histograms per class",
        "  • operating_points.png        — Sensitivity/Specificity trade-off",
        "",
        "═" * 65,
    ]

    report = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    return report


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path = "artifacts/clinical_eval",
    device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
    """Run the full clinical evaluation pipeline.

    Returns a dictionary of all computed metrics and saves plots + report
    to *output_dir*.
    """
    _apply_style()

    device = select_device(device)
    checkpoint_path = Path(checkpoint_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model, meta = load_checkpoint(checkpoint_path, device=device)
    class_names = list(meta["class_names"])
    image_size = int(meta["image_size"])
    pneumonia_index = class_names.index("PNEUMONIA") if "PNEUMONIA" in class_names else 1

    # Load data
    print(f"Loading data from: {data_dir}")
    data_root = resolve_data_root(data_dir)
    loaders, _ = make_dataloaders(
        data_root=str(data_root),
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        num_workers=num_workers,
    )

    # Collect predictions on test set
    print("Collecting predictions on test set...")
    labels, probs = collect_predictions(model, loaders["test"], device, pneumonia_index)
    print(f"  Samples: {len(labels)} (pneumonia={labels.sum()}, normal={(1-labels).sum()})")

    # Generate all plots and metrics
    print("\nGenerating evaluation plots...")

    print("  → ROC curve")
    roc_results = plot_roc_curve(labels, probs, out / "roc_curve.png")

    print("  → Precision-Recall curve")
    pr_results = plot_precision_recall_curve(labels, probs, out / "precision_recall_curve.png")

    print("  → Confusion matrix")
    cm_results = plot_confusion_matrix(labels, probs, 0.5, out / "confusion_matrix.png")

    print("  → Calibration curve")
    cal_results = plot_calibration_curve(labels, probs, out / "calibration_curve.png")

    print("  → Confidence distribution")
    plot_confidence_distribution(labels, probs, out / "confidence_distribution.png")

    print("  → Operating point analysis")
    ops_results = plot_operating_points(labels, probs, out / "operating_points.png")

    print("  → Bootstrap confidence intervals (1000 iterations)...")
    boot_results = bootstrap_metrics(labels, probs)

    # Assemble all results
    all_results = {
        "roc": roc_results,
        "precision_recall": pr_results,
        "confusion_matrix": cm_results,
        "calibration": cal_results,
        "operating_points": ops_results,
        "bootstrap": boot_results,
        "checkpoint": str(checkpoint_path.resolve()),
        "n_samples": int(len(labels)),
        "n_pneumonia": int(labels.sum()),
        "n_normal": int((1 - labels).sum()),
    }

    # Save JSON results
    json_path = out / "evaluation_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Results saved to: {json_path}")

    # Generate and print summary report
    print()
    report = generate_summary_report(all_results, out / "evaluation_report.txt")
    print(report)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clinical evaluation on pneumonia model.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument(
        "--data-dir",
        default="data/chest-xray-pneumonia/chest_xray",
        help="Path to directory containing train/val/test splits.",
    )
    parser.add_argument("--output-dir", default="artifacts/clinical_eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
