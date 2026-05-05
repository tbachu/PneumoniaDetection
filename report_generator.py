"""Structured radiology-style report generator.

Takes a ClinicalAnalysis or MultiLabelAnalysis result and formats it
as a clinical report suitable for radiologist review.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from clinical_engine import (
    ClinicalAnalysis,
    MultiLabelAnalysis,
    SeverityLevel,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_INDICATOR: dict[SeverityLevel, str] = {
    SeverityLevel.NEGATIVE: "●○○○○",
    SeverityLevel.LOW:      "●●○○○",
    SeverityLevel.MODERATE: "●●●○○",
    SeverityLevel.HIGH:     "●●●●○",
    SeverityLevel.CRITICAL: "●●●●●",
}

_RECOMMENDATIONS: dict[SeverityLevel, list[str]] = {
    SeverityLevel.NEGATIVE: [
        "No acute pulmonary findings detected.",
        "Routine follow-up as clinically indicated.",
    ],
    SeverityLevel.LOW: [
        "Equivocal findings — clinical correlation recommended.",
        "Consider repeat imaging if symptoms persist.",
    ],
    SeverityLevel.MODERATE: [
        "Findings suggestive of pathology.",
        "Recommend clinical correlation with symptoms and labs.",
        "Consider follow-up imaging in 24–48 hours.",
    ],
    SeverityLevel.HIGH: [
        "Findings consistent with pathology.",
        "Recommend clinical correlation with patient symptoms.",
        "Consider follow-up imaging if clinically indicated.",
        "Review Grad-CAM overlay for region localization.",
    ],
    SeverityLevel.CRITICAL: [
        "⚠ High-confidence pathology detected — urgent review recommended.",
        "Immediate clinical correlation required.",
        "Consider CT chest for further characterization.",
        "Review Grad-CAM overlay for region localization.",
    ],
}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    analysis: ClinicalAnalysis | MultiLabelAnalysis,
    patient_id: str = "ANONYMOUS",
    study_date: str | None = None,
) -> str:
    """Return a plain-text structured radiology report.

    Dispatches to binary or multi-label format based on the analysis type.
    """
    if isinstance(analysis, MultiLabelAnalysis):
        return _generate_multilabel_report(analysis, patient_id, study_date)
    return _generate_binary_report(analysis, patient_id, study_date)


def _generate_binary_report(
    analysis: ClinicalAnalysis,
    patient_id: str = "ANONYMOUS",
    study_date: str | None = None,
) -> str:
    """Generate report for binary (PNEUMONIA / NORMAL) analysis."""
    if study_date is None:
        study_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    sev = analysis.severity
    unc = analysis.uncertainty
    reg = analysis.regions
    met = analysis.clinical_metrics
    mod = analysis.model_info

    # Uncertainty narrative
    if unc.std_probability < 0.05:
        unc_desc = "low — model is confident"
    elif unc.std_probability < 0.10:
        unc_desc = "moderate — consider clinical correlation"
    else:
        unc_desc = "HIGH — model is uncertain, senior review recommended"

    # Recommendations
    recs = list(_RECOMMENDATIONS.get(sev.level, _RECOMMENDATIONS[SeverityLevel.MODERATE]))
    if unc.is_uncertain:
        recs.append("⚠ Model uncertainty is elevated — findings should be independently verified.")
    rec_lines = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(recs))

    # Severity bar
    sev_bar = _SEVERITY_INDICATOR.get(sev.level, "○○○○○")

    # Region zone scores (sorted by activation)
    zone_lines = "\n".join(
        f"    {name}: {score:.1%}"
        for name, score in sorted(reg.region_scores.items(), key=lambda x: -x[1])
    )

    # Metrics section — only show when real values have been computed
    has_metrics = met.get("sensitivity", 0) > 0
    if has_metrics:
        metrics_block = (
            f"  Sensitivity:       {met['sensitivity']:.1%}\n"
            f"  Specificity:       {met['specificity']:.1%}\n"
            f"  PPV:               {met['ppv']:.1%}\n"
            f"  NPV:               {met['npv']:.1%}"
        )
    else:
        metrics_block = "  Run calibrate.py to compute clinical performance metrics."

    report = f"""\
═══════════════════════════════════════════════════════════
              CHEST X-RAY ANALYSIS REPORT
                AI-Assisted Findings
═══════════════════════════════════════════════════════════

  Patient ID:        {patient_id}
  Study Date:        {study_date}
  Analysis Model:    {mod['model_name']} (temp={mod['temperature']:.2f})

─── CLASSIFICATION ────────────────────────────────────────
  Result:            {analysis.prediction}
  Severity:          {sev.label} ({sev.score}/5) {sev_bar}
  Description:       {sev.description}

─── CONFIDENCE ────────────────────────────────────────────
  Pneumonia Prob:    {analysis.pneumonia_probability:.1%}
  Normal Prob:       {analysis.normal_probability:.1%}
  Calibrated:        {analysis.calibrated_probability:.1%}
  Uncertainty:       ±{unc.std_probability:.1%} ({unc_desc})
  MC Samples:        {unc.num_samples}

─── REGION ANALYSIS ───────────────────────────────────────
  Primary Region:    {reg.primary_region}
  Laterality:        {reg.laterality}
  Affected Area:     ~{reg.affected_area_pct:.1f}% of lung field
  Pattern:           {reg.pattern}
  Zone Activation:
{zone_lines}

─── MODEL PERFORMANCE (threshold={met.get('threshold', 0.5)}) ─────────
{metrics_block}

─── RECOMMENDATION ────────────────────────────────────────
{rec_lines}

─── DISCLAIMER ────────────────────────────────────────────
  This is an AI-assisted analysis for clinical decision
  support only. All findings must be reviewed and confirmed
  by a qualified radiologist. This tool is not intended as
  a standalone diagnostic device and has not been cleared
  by the FDA for clinical use.
═══════════════════════════════════════════════════════════"""

    return report


def _generate_multilabel_report(
    analysis: MultiLabelAnalysis,
    patient_id: str = "ANONYMOUS",
    study_date: str | None = None,
) -> str:
    """Generate report for multi-label (14 finding) analysis."""
    if study_date is None:
        study_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    sev = analysis.severity
    unc = analysis.uncertainty
    reg = analysis.regions
    mod = analysis.model_info

    # Overall uncertainty narrative
    max_std = unc.get("max_std", 0.0) if isinstance(unc, dict) else 0.0
    if max_std < 0.05:
        unc_desc = "low — model is confident"
    elif max_std < 0.10:
        unc_desc = "moderate — consider clinical correlation"
    else:
        unc_desc = "HIGH — model is uncertain, senior review recommended"

    # Severity bar
    sev_bar = _SEVERITY_INDICATOR.get(sev.level, "○○○○○")

    # Positive findings table
    positive = [f for f in analysis.findings if f["positive"]]
    negative = [f for f in analysis.findings if not f["positive"]]

    if positive:
        finding_lines = []
        for f in positive:
            unc_val = f.get("uncertainty", 0)
            finding_lines.append(
                f"  ▸ {f['display_name']:<22s} {f['probability']:.1%}  (±{unc_val:.1%})"
            )
        findings_block = "\n".join(finding_lines)
    else:
        findings_block = "  No significant findings detected."

    # Negative findings (summarized)
    neg_names = [f["display_name"] for f in negative[:6]]
    neg_summary = ", ".join(neg_names)
    if len(negative) > 6:
        neg_summary += f", +{len(negative) - 6} more"

    # Region zone scores
    zone_lines = "\n".join(
        f"    {name}: {score:.1%}"
        for name, score in sorted(reg.region_scores.items(), key=lambda x: -x[1])
    )

    # Recommendations
    recs = list(_RECOMMENDATIONS.get(sev.level, _RECOMMENDATIONS[SeverityLevel.MODERATE]))
    is_uncertain = unc.get("is_uncertain", False) if isinstance(unc, dict) else False
    if is_uncertain:
        recs.append("⚠ Model uncertainty is elevated — findings should be independently verified.")
    rec_lines = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(recs))

    gradcam_target = mod.get("gradcam_target", "top finding")

    report = f"""\
═══════════════════════════════════════════════════════════
              CHEST X-RAY ANALYSIS REPORT
          Multi-Finding AI-Assisted Analysis
═══════════════════════════════════════════════════════════

  Patient ID:        {patient_id}
  Study Date:        {study_date}
  Analysis Model:    {mod['model_name']} ({mod.get('num_classes', 14)}-class multi-label)
  Dataset:           NIH ChestX-ray14

─── OVERALL ASSESSMENT ────────────────────────────────────
  Result:            {analysis.prediction}
  Severity:          {sev.label} ({sev.score}/5) {sev_bar}
  Description:       {sev.description}
  Uncertainty:       ±{max_std:.1%} ({unc_desc})

─── POSITIVE FINDINGS ─────────────────────────────────────
{findings_block}

─── NEGATIVE FINDINGS ─────────────────────────────────────
  {neg_summary}

─── REGION ANALYSIS (Grad-CAM for: {gradcam_target}) ──────
  Primary Region:    {reg.primary_region}
  Laterality:        {reg.laterality}
  Affected Area:     ~{reg.affected_area_pct:.1f}% of lung field
  Pattern:           {reg.pattern}
  Zone Activation:
{zone_lines}

─── RECOMMENDATION ────────────────────────────────────────
{rec_lines}

─── DISCLAIMER ────────────────────────────────────────────
  This is an AI-assisted analysis for clinical decision
  support only. All findings must be reviewed and confirmed
  by a qualified radiologist. This tool is not intended as
  a standalone diagnostic device and has not been cleared
  by the FDA for clinical use.
═══════════════════════════════════════════════════════════"""

    return report


# ---------------------------------------------------------------------------
# JSON-serializable output
# ---------------------------------------------------------------------------

def analysis_to_dict(analysis: ClinicalAnalysis | MultiLabelAnalysis) -> dict[str, Any]:
    """Convert an analysis result to a JSON-serializable dictionary."""
    if isinstance(analysis, MultiLabelAnalysis):
        return _multilabel_to_dict(analysis)
    return _binary_to_dict(analysis)


def _binary_to_dict(analysis: ClinicalAnalysis) -> dict[str, Any]:
    """Convert a binary ClinicalAnalysis to a JSON-serializable dictionary."""
    return {
        "task": "binary",
        "prediction": analysis.prediction,
        "pneumonia_probability": analysis.pneumonia_probability,
        "normal_probability": analysis.normal_probability,
        "calibrated_probability": analysis.calibrated_probability,
        "severity": {
            "level": analysis.severity.score,
            "label": analysis.severity.label,
            "description": analysis.severity.description,
        },
        "uncertainty": {
            "mean_probability": round(analysis.uncertainty.mean_probability, 4),
            "std_probability": round(analysis.uncertainty.std_probability, 4),
            "is_uncertain": analysis.uncertainty.is_uncertain,
            "num_samples": analysis.uncertainty.num_samples,
        },
        "regions": {
            "primary_region": analysis.regions.primary_region,
            "laterality": analysis.regions.laterality,
            "affected_area_pct": analysis.regions.affected_area_pct,
            "pattern": analysis.regions.pattern,
            "zone_scores": analysis.regions.region_scores,
        },
        "model_info": analysis.model_info,
        "clinical_metrics": analysis.clinical_metrics,
        "gradcam_overlay_b64": analysis.gradcam_overlay_b64,
        "gradcam_raw_b64": analysis.gradcam_raw_b64,
    }


def _multilabel_to_dict(analysis: MultiLabelAnalysis) -> dict[str, Any]:
    """Convert a MultiLabelAnalysis to a JSON-serializable dictionary."""
    unc = analysis.uncertainty

    return {
        "task": "multilabel",
        "prediction": analysis.prediction,
        "findings": analysis.findings,
        "positive_findings": analysis.positive_findings,
        "pneumonia_probability": max(
            (f["probability"] for f in analysis.findings), default=0.0
        ),
        "severity": {
            "level": analysis.severity.score,
            "label": analysis.severity.label,
            "description": analysis.severity.description,
        },
        "uncertainty": {
            "mean_probability": unc.get("means", [0])[0] if isinstance(unc, dict) else 0,
            "std_probability": unc.get("max_std", 0) if isinstance(unc, dict) else 0,
            "is_uncertain": unc.get("is_uncertain", False) if isinstance(unc, dict) else False,
            "num_samples": unc.get("num_samples", 0) if isinstance(unc, dict) else 0,
        },
        "regions": {
            "primary_region": analysis.regions.primary_region,
            "laterality": analysis.regions.laterality,
            "affected_area_pct": analysis.regions.affected_area_pct,
            "pattern": analysis.regions.pattern,
            "zone_scores": analysis.regions.region_scores,
        },
        "model_info": analysis.model_info,
        "clinical_metrics": analysis.clinical_metrics,
        "gradcam_overlay_b64": analysis.gradcam_overlay_b64,
        "gradcam_raw_b64": analysis.gradcam_raw_b64,
    }
