"""Structured radiology-style report generator.

Takes a ClinicalAnalysis result and formats it as a clinical report
suitable for radiologist review.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from clinical_engine import ClinicalAnalysis, SeverityLevel


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
        "Findings suggestive of pneumonia.",
        "Recommend clinical correlation with symptoms and labs.",
        "Consider follow-up imaging in 24–48 hours.",
    ],
    SeverityLevel.HIGH: [
        "Findings consistent with pneumonia.",
        "Recommend clinical correlation with patient symptoms.",
        "Consider follow-up imaging if clinically indicated.",
        "Review Grad-CAM overlay for region localization.",
    ],
    SeverityLevel.CRITICAL: [
        "⚠ High-confidence pneumonia detected — urgent review recommended.",
        "Immediate clinical correlation required.",
        "Consider CT chest for further characterization.",
        "Review Grad-CAM overlay for region localization.",
    ],
}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    analysis: ClinicalAnalysis,
    patient_id: str = "ANONYMOUS",
    study_date: str | None = None,
) -> str:
    """Return a plain-text structured radiology report.

    Parameters
    ----------
    analysis : ClinicalAnalysis
        Output of ``ClinicalAnalyzer.analyze()``.
    patient_id : str
        Optional patient identifier for the report header.
    study_date : str | None
        Optional date string; defaults to current date/time.
    """
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


# ---------------------------------------------------------------------------
# JSON-serializable output
# ---------------------------------------------------------------------------

def analysis_to_dict(analysis: ClinicalAnalysis) -> dict[str, Any]:
    """Convert a *ClinicalAnalysis* to a JSON-serializable dictionary.

    The Grad-CAM base64 fields are preserved so the consuming application
    can embed them directly in ``<img>`` tags.
    """
    return {
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
