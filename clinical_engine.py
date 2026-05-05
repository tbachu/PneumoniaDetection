"""Clinical inference engine for chest X-ray analysis.

Wraps the trained classifier with clinically meaningful output:
- 5-level severity scoring
- Calibrated confidence via temperature scaling
- Uncertainty estimation via Monte Carlo Dropout
- Lung region analysis from Grad-CAM heatmaps

Supports both **binary** (NORMAL / PNEUMONIA) and **multi-label**
(14 pathology) modes via the ``task`` parameter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from modeling import (
    GradCAM,
    blend_heatmap,
    disable_inplace_relu,
    encode_pil_to_base64,
    get_target_layer,
    heatmap_to_pil,
    preprocess_pil_image,
)


# ---------------------------------------------------------------------------
# Constants — multi-label finding names
# ---------------------------------------------------------------------------

NIH_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# Human-readable names for display
_FINDING_DISPLAY = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Edema": "Pulmonary Edema",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Pleural_Thickening": "Pleural Thickening",
    "Hernia": "Hernia",
}


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

class SeverityLevel(IntEnum):
    """Five-level severity scale for chest X-ray findings."""
    NEGATIVE = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5


# (upper_bound, level, description)
_SEVERITY_THRESHOLDS: list[tuple[float, SeverityLevel, str]] = [
    (0.20, SeverityLevel.NEGATIVE, "No significant findings"),
    (0.40, SeverityLevel.LOW, "Minimal/equivocal findings — consider clinical correlation"),
    (0.60, SeverityLevel.MODERATE, "Findings suggestive of pathology — recommend follow-up"),
    (0.80, SeverityLevel.HIGH, "Findings consistent with pathology"),
    (1.01, SeverityLevel.CRITICAL, "High-confidence findings — urgent review recommended"),
]


@dataclass(frozen=True)
class SeverityAssessment:
    level: SeverityLevel
    score: int            # 1-5
    label: str            # e.g. "HIGH"
    description: str
    pneumonia_probability: float


def assess_severity(probability: float) -> SeverityAssessment:
    """Map a probability [0, 1] to a severity assessment."""
    for threshold, level, description in _SEVERITY_THRESHOLDS:
        if probability <= threshold:
            return SeverityAssessment(
                level=level,
                score=int(level),
                label=level.name,
                description=description,
                pneumonia_probability=probability,
            )
    # Fallback
    return SeverityAssessment(
        level=SeverityLevel.CRITICAL,
        score=5,
        label="CRITICAL",
        description=_SEVERITY_THRESHOLDS[-1][2],
        pneumonia_probability=probability,
    )


def assess_severity_multilabel(
    findings: list[dict[str, Any]],
    max_prob: float,
) -> SeverityAssessment:
    """Assess overall severity from multiple findings.

    Uses the maximum probability and number of positive findings to
    determine the overall severity level.
    """
    n_positive = sum(1 for f in findings if f["positive"])

    # Adjust severity: many findings at moderate confidence = still serious
    effective_prob = max_prob
    if n_positive >= 3:
        effective_prob = min(1.0, max_prob + 0.15)
    elif n_positive >= 2:
        effective_prob = min(1.0, max_prob + 0.05)

    sev = assess_severity(effective_prob)

    # Override description for multi-label
    if n_positive == 0:
        desc = "No significant findings detected"
    elif n_positive == 1:
        desc = f"1 finding detected (max confidence {max_prob:.0%})"
    else:
        desc = f"{n_positive} findings detected (max confidence {max_prob:.0%})"

    return SeverityAssessment(
        level=sev.level,
        score=sev.score,
        label=sev.label,
        description=desc,
        pneumonia_probability=max_prob,
    )


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Post-hoc temperature scaling for probability calibration.

    After training, learn a single scalar *temperature* on the validation set
    so that the softmax outputs are better calibrated (i.e. a reported 80%
    confidence actually means correct ~80% of the time).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Learn optimal temperature on a held-out validation set.

        Returns the learned temperature value.
        """
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits_tensor / temperature
            loss = F.cross_entropy(scaled, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(_closure)
        self.temperature = max(temperature.item(), 0.1)  # clamp to avoid division issues
        return self.temperature

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to raw logits."""
        return logits / self.temperature


# ---------------------------------------------------------------------------
# Uncertainty estimation (Monte Carlo Dropout)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UncertaintyEstimate:
    mean_probability: float
    std_probability: float
    is_uncertain: bool      # True when std exceeds threshold
    num_samples: int
    raw_probabilities: list[float]

# Standard deviation above which the model's prediction is flagged as uncertain
_UNCERTAINTY_THRESHOLD = 0.10


def mc_dropout_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_samples: int = 30,
    pneumonia_index: int = 1,
) -> UncertaintyEstimate:
    """Run *num_samples* stochastic forward passes with dropout enabled.

    Returns the mean and standard deviation of the pneumonia probability,
    plus a flag indicating whether the model is uncertain.
    """
    model.train()  # activates dropout

    probabilities: list[float] = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            probabilities.append(probs[pneumonia_index].item())

    model.eval()  # restore

    arr = np.array(probabilities)
    mean_p = float(arr.mean())
    std_p = float(arr.std())

    return UncertaintyEstimate(
        mean_probability=mean_p,
        std_probability=std_p,
        is_uncertain=std_p > _UNCERTAINTY_THRESHOLD,
        num_samples=num_samples,
        raw_probabilities=probabilities,
    )


def mc_dropout_multilabel(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_samples: int = 30,
    num_classes: int = 14,
) -> dict[str, Any]:
    """MC Dropout for multi-label: returns mean and std per finding."""
    model.train()

    all_probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            all_probs.append(probs)

    model.eval()
    all_probs_arr = np.array(all_probs)  # (num_samples, num_classes)
    means = all_probs_arr.mean(axis=0)
    stds = all_probs_arr.std(axis=0)

    # Overall uncertainty = max std across positive findings (or all)
    max_std = float(stds.max())

    return {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "max_std": max_std,
        "is_uncertain": max_std > _UNCERTAINTY_THRESHOLD,
        "num_samples": num_samples,
    }


# ---------------------------------------------------------------------------
# Region analysis from Grad-CAM
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionAnalysis:
    primary_region: str         # e.g. "Right lower zone"
    laterality: str             # "Unilateral (right)", "Bilateral", etc.
    affected_area_pct: float    # % of lung field with significant activation
    pattern: str                # "Focal consolidation", "Diffuse", etc.
    region_scores: dict[str, float]


def analyze_regions(
    heatmap: np.ndarray,
    activation_threshold: float = 0.3,
) -> RegionAnalysis:
    """Derive lung-region information from a Grad-CAM heatmap.

    Splits the heatmap into four quadrants approximating upper/lower and
    left/right lung zones. Uses *anatomical* convention for a standard PA
    chest X-ray (patient's right lung = left side of image).
    """
    h, w = heatmap.shape
    mid_w = w // 2
    mid_h = h // 2

    # Anatomical labelling (PA view: patient right = image left)
    regions = {
        "Right upper zone": heatmap[:mid_h, :mid_w],
        "Right lower zone": heatmap[mid_h:, :mid_w],
        "Left upper zone":  heatmap[:mid_h, mid_w:],
        "Left lower zone":  heatmap[mid_h:, mid_w:],
    }

    region_scores = {name: float(r.mean()) for name, r in regions.items()}
    primary_region = max(region_scores, key=lambda k: region_scores[k])

    # Laterality
    right_activation = float(heatmap[:, :mid_w].mean())
    left_activation = float(heatmap[:, mid_w:].mean())
    total_activation = float(heatmap.mean())

    if total_activation < 0.05:
        laterality = "No significant lateralization"
    elif right_activation > 2.0 * left_activation:
        laterality = "Unilateral (right)"
    elif left_activation > 2.0 * right_activation:
        laterality = "Unilateral (left)"
    else:
        laterality = "Bilateral"

    # Affected area (% of pixels above threshold)
    activated = (heatmap > activation_threshold).sum()
    affected_area_pct = float(activated / heatmap.size * 100)

    # Pattern (spatial spread of high-activation pixels)
    high_activation = heatmap > 0.6
    if high_activation.sum() == 0:
        pattern = "No significant focal activation"
    else:
        coords = np.argwhere(high_activation)
        spread_h = (coords[:, 0].max() - coords[:, 0].min()) / h
        spread_w = (coords[:, 1].max() - coords[:, 1].min()) / w
        spread = max(spread_h, spread_w)
        if spread < 0.3:
            pattern = "Focal consolidation"
        elif spread < 0.6:
            pattern = "Patchy / multifocal"
        else:
            pattern = "Diffuse"

    return RegionAnalysis(
        primary_region=primary_region,
        laterality=laterality,
        affected_area_pct=round(affected_area_pct, 1),
        pattern=pattern,
        region_scores=region_scores,
    )


# ---------------------------------------------------------------------------
# Full clinical analysis pipeline
# ---------------------------------------------------------------------------

# Placeholder metrics — replaced once `calibrate.py` is run on the test set.
_DEFAULT_CLINICAL_METRICS: dict[str, float] = {
    "sensitivity": 0.0,
    "specificity": 0.0,
    "ppv": 0.0,
    "npv": 0.0,
    "threshold": 0.5,
    "auc_roc": 0.0,
}


@dataclass(frozen=True)
class ClinicalAnalysis:
    """Complete clinical analysis result for a single chest X-ray."""
    prediction: str
    pneumonia_probability: float
    normal_probability: float
    calibrated_probability: float
    severity: SeverityAssessment
    uncertainty: UncertaintyEstimate
    regions: RegionAnalysis
    gradcam_overlay_b64: str
    gradcam_raw_b64: str
    model_info: dict[str, Any]
    clinical_metrics: dict[str, float]


@dataclass(frozen=True)
class MultiLabelAnalysis:
    """Complete clinical analysis result for multi-label detection."""
    prediction: str                  # Top finding or "No Finding"
    findings: list[dict[str, Any]]   # All 14 findings with probabilities
    positive_findings: list[str]     # Names of positive findings
    severity: SeverityAssessment
    uncertainty: dict[str, Any]      # MC Dropout per-finding
    regions: RegionAnalysis          # For the top finding
    gradcam_overlay_b64: str
    gradcam_raw_b64: str
    model_info: dict[str, Any]
    clinical_metrics: dict[str, Any]


class ClinicalAnalyzer:
    """End-to-end clinical analysis wrapping a trained chest X-ray model.

    Supports both binary (PNEUMONIA vs NORMAL) and multi-label (14 finding)
    models. The ``task`` parameter controls the behavior.

    Usage::

        model, meta = load_checkpoint("artifacts/improved_resnet18.pt")
        analyzer = ClinicalAnalyzer(model, **meta_kwargs)
        analyzer.load_calibration("artifacts/improved_resnet18_calibration.json")
        result = analyzer.analyze(pil_image)
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        class_names: list[str],
        image_size: int,
        device: str,
        task: str = "binary",
        temperature: float = 1.0,
        mc_samples: int = 30,
        clinical_metrics: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.class_names = class_names
        self.image_size = image_size
        self.device = device
        self.task = task  # "binary" or "multilabel"
        self.temperature_scaler = TemperatureScaler(temperature)
        self.mc_samples = mc_samples
        self.clinical_metrics = clinical_metrics or _DEFAULT_CLINICAL_METRICS.copy()

        if self.task == "binary":
            # Resolve class indices
            self.pneumonia_index = (
                self.class_names.index("PNEUMONIA") if "PNEUMONIA" in self.class_names else 1
            )
            self.normal_index = 1 - self.pneumonia_index
        else:
            # Multi-label: class_names are finding names
            self.pneumonia_index = (
                self.class_names.index("Pneumonia") if "Pneumonia" in self.class_names else 6
            )

    # -- calibration helpers ------------------------------------------------

    def calibrate(self, val_loader: DataLoader) -> float:
        """Calibrate temperature on a validation loader. Returns temperature."""
        return self.temperature_scaler.calibrate(self.model, val_loader, self.device)

    def load_calibration(self, path: str | Path) -> None:
        """Load pre-computed calibration (temperature + metrics) from JSON."""
        data = json.loads(Path(path).read_text())
        self.temperature_scaler.temperature = float(data.get("temperature", 1.0))
        if "metrics" in data:
            self.clinical_metrics.update(data["metrics"])

    def save_calibration(self, path: str | Path) -> None:
        """Persist calibration state to JSON."""
        payload = {
            "temperature": self.temperature_scaler.temperature,
            "metrics": self.clinical_metrics,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2))

    # -- main analysis ------------------------------------------------------

    def analyze(self, image: Image.Image) -> ClinicalAnalysis | MultiLabelAnalysis:
        """Run full clinical analysis on a single chest X-ray image.

        Dispatches to binary or multi-label analysis based on self.task.
        """
        if self.task == "multilabel":
            return self._analyze_multilabel(image)
        return self._analyze_binary(image)

    def _analyze_binary(self, image: Image.Image) -> ClinicalAnalysis:
        """Binary analysis: PNEUMONIA vs NORMAL."""
        input_tensor = preprocess_pil_image(image, self.image_size).to(self.device)

        # 1. Standard inference with temperature-calibrated probabilities
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            calibrated_logits = self.temperature_scaler.scale(logits)
            probs = torch.softmax(calibrated_logits, dim=1)[0].cpu().numpy()

        pneumonia_prob = float(probs[self.pneumonia_index])
        normal_prob = float(probs[self.normal_index])
        prediction = "PNEUMONIA" if pneumonia_prob > 0.5 else "NORMAL"

        # 2. Severity assessment
        severity = assess_severity(pneumonia_prob)

        # 3. Uncertainty via MC Dropout
        uncertainty = mc_dropout_inference(
            self.model,
            input_tensor,
            num_samples=self.mc_samples,
            pneumonia_index=self.pneumonia_index,
        )

        # 4. Grad-CAM region analysis
        self.model.eval()
        disable_inplace_relu(self.model)
        target_layer = get_target_layer(self.model, self.model_name)
        cam = GradCAM(self.model, target_layer)
        try:
            heatmap, _ = cam.generate(input_tensor, class_index=self.pneumonia_index)
        finally:
            cam.close()

        regions = analyze_regions(heatmap)

        # 5. Encode visualizations
        overlay = blend_heatmap(image, heatmap, alpha=0.45)
        overlay_b64 = encode_pil_to_base64(overlay)
        raw_b64 = encode_pil_to_base64(heatmap_to_pil(heatmap))

        return ClinicalAnalysis(
            prediction=prediction,
            pneumonia_probability=round(pneumonia_prob, 4),
            normal_probability=round(normal_prob, 4),
            calibrated_probability=round(pneumonia_prob, 4),
            severity=severity,
            uncertainty=uncertainty,
            regions=regions,
            gradcam_overlay_b64=overlay_b64,
            gradcam_raw_b64=raw_b64,
            model_info={
                "model_name": self.model_name,
                "image_size": self.image_size,
                "temperature": self.temperature_scaler.temperature,
                "mc_samples": self.mc_samples,
            },
            clinical_metrics=self.clinical_metrics,
        )

    def _analyze_multilabel(self, image: Image.Image) -> MultiLabelAnalysis:
        """Multi-label analysis: 14 chest pathologies."""
        input_tensor = preprocess_pil_image(image, self.image_size).to(self.device)
        threshold = 0.5

        # 1. Inference with sigmoid probabilities
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            scaled_logits = self.temperature_scaler.scale(logits)
            probs = torch.sigmoid(scaled_logits)[0].cpu().numpy()

        # 2. Build per-finding results
        findings: list[dict[str, Any]] = []
        positive_names: list[str] = []

        for i, name in enumerate(self.class_names):
            prob = float(probs[i])
            positive = prob >= threshold
            if positive:
                positive_names.append(name)
            findings.append({
                "name": name,
                "display_name": _FINDING_DISPLAY.get(name, name),
                "probability": round(prob, 4),
                "positive": positive,
            })

        # Sort by probability descending
        findings.sort(key=lambda f: f["probability"], reverse=True)

        # Top finding for prediction label
        if positive_names:
            top_finding = findings[0]["display_name"]
            prediction = f"{top_finding} (+{len(positive_names) - 1} more)" if len(positive_names) > 1 else top_finding
        else:
            prediction = "No Finding"

        # Max probability for severity
        max_prob = float(probs.max())

        # 3. Severity based on overall picture
        severity = assess_severity_multilabel(findings, max_prob)

        # 4. MC Dropout uncertainty (per-finding)
        uncertainty_data = mc_dropout_multilabel(
            self.model,
            input_tensor,
            num_samples=self.mc_samples,
            num_classes=len(self.class_names),
        )

        # Attach per-finding uncertainties
        for i, name in enumerate(self.class_names):
            for f in findings:
                if f["name"] == name:
                    f["uncertainty"] = round(uncertainty_data["stds"][i], 4)

        # Build an UncertaintyEstimate-like for the overall result
        # (Use the top finding's uncertainty for backward compat)
        top_idx = int(np.argmax(probs))
        uncertainty_compat = UncertaintyEstimate(
            mean_probability=float(uncertainty_data["means"][top_idx]),
            std_probability=float(uncertainty_data["stds"][top_idx]),
            is_uncertain=uncertainty_data["is_uncertain"],
            num_samples=uncertainty_data["num_samples"],
            raw_probabilities=[],
        )

        # 5. Grad-CAM for the top positive finding (or highest-prob finding)
        self.model.eval()
        disable_inplace_relu(self.model)
        target_layer = get_target_layer(self.model, self.model_name)
        cam = GradCAM(self.model, target_layer)
        try:
            gradcam_class = int(np.argmax(probs))
            heatmap, _ = cam.generate(input_tensor, class_index=gradcam_class)
        finally:
            cam.close()

        regions = analyze_regions(heatmap)

        # 6. Encode visualizations
        overlay = blend_heatmap(image, heatmap, alpha=0.45)
        overlay_b64 = encode_pil_to_base64(overlay)
        raw_b64 = encode_pil_to_base64(heatmap_to_pil(heatmap))

        return MultiLabelAnalysis(
            prediction=prediction,
            findings=findings,
            positive_findings=positive_names,
            severity=severity,
            uncertainty=uncertainty_data,
            regions=regions,
            gradcam_overlay_b64=overlay_b64,
            gradcam_raw_b64=raw_b64,
            model_info={
                "model_name": self.model_name,
                "image_size": self.image_size,
                "task": "multilabel",
                "temperature": self.temperature_scaler.temperature,
                "mc_samples": self.mc_samples,
                "num_classes": len(self.class_names),
                "gradcam_target": self.class_names[int(np.argmax(probs))],
            },
            clinical_metrics=self.clinical_metrics,
        )
