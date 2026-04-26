"""Calibrate temperature scaling and compute clinical metrics.

Run this once after training to:
1. Learn the optimal temperature parameter on the validation set.
2. Compute sensitivity, specificity, PPV, NPV, and AUC on the test set.
3. Save everything to a JSON sidecar file alongside the checkpoint.

Usage::

    python calibrate.py \\
        --checkpoint artifacts/improved_resnet18.pt \\
        --data-dir data/chest-xray-pneumonia/chest_xray \\
        --device mps --num-workers 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader

from clinical_engine import ClinicalAnalyzer
from modeling import (
    load_checkpoint,
    make_dataloaders,
    resolve_data_root,
    select_device,
)


def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    pneumonia_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on a loader, return (labels, pred_indices, pneumonia_probs)."""
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
    preds_arr = (probs_arr >= 0.5).astype(int)
    return labels_arr, preds_arr, probs_arr


def _compute_clinical_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    pneumonia_index: int,
) -> dict[str, float]:
    """Compute sensitivity, specificity, PPV, NPV, AUC-ROC."""
    # Binarise labels relative to pneumonia
    binary_labels = (labels == pneumonia_index).astype(int)
    binary_preds = preds  # already 0/1 based on prob >= 0.5

    cm = confusion_matrix(binary_labels, binary_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    fpr, tpr, _ = roc_curve(binary_labels, probs)
    auc_roc = float(auc(fpr, tpr))

    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average="binary", zero_division=0,
    )

    return {
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "auc_roc": round(auc_roc, 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "threshold": 0.5,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate model and compute clinical metrics.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument(
        "--data-dir",
        default="data/chest-xray-pneumonia/chest_xray",
        help="Path to directory containing train/val/test splits.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <checkpoint>_calibration.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, meta = load_checkpoint(checkpoint_path, device=device)
    image_size = int(meta["image_size"])
    class_names = list(meta["class_names"])
    model_name = str(meta["model_name"])

    pneumonia_index = class_names.index("PNEUMONIA") if "PNEUMONIA" in class_names else 1

    print(f"Resolving data root: {args.data_dir}")
    data_root = resolve_data_root(args.data_dir)

    loaders, _ = make_dataloaders(
        data_root=str(data_root),
        image_size=image_size,
        batch_size=args.batch_size,
        augment=False,
        num_workers=args.num_workers,
    )

    # --- Temperature calibration on val set --------------------------------
    print("\n── Temperature Calibration (validation set) ──")
    analyzer = ClinicalAnalyzer(
        model=model,
        model_name=model_name,
        class_names=class_names,
        image_size=image_size,
        device=device,
    )
    temperature = analyzer.calibrate(loaders["val"])
    print(f"Learned temperature: {temperature:.4f}")

    # --- Clinical metrics on test set --------------------------------------
    print("\n── Clinical Metrics (test set) ──")
    labels, preds, probs = _collect_predictions(model, loaders["test"], device, pneumonia_index)
    metrics = _compute_clinical_metrics(labels, preds, probs, pneumonia_index)

    print(f"  Sensitivity : {metrics['sensitivity']:.1%}")
    print(f"  Specificity : {metrics['specificity']:.1%}")
    print(f"  PPV         : {metrics['ppv']:.1%}")
    print(f"  NPV         : {metrics['npv']:.1%}")
    print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")

    # --- Save calibration --------------------------------------------------
    output_path = Path(args.output) if args.output else checkpoint_path.with_name(
        checkpoint_path.stem + "_calibration.json"
    )
    calibration = {
        "temperature": round(temperature, 6),
        "metrics": metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "model_name": model_name,
        "class_names": class_names,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=2))
    print(f"\nCalibration saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
