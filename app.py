from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from clinical_engine import ClinicalAnalyzer
from modeling import (
    encode_pil_to_base64,
    generate_gradcam_overlay,
    heatmap_to_pil,
    load_checkpoint,
    predict_image,
    select_device,
)
from report_generator import analysis_to_dict, generate_report

app = FastAPI(title="Chest X-Ray Clinical Analysis API", version="2.0.0")


@app.on_event("startup")
def load_model_on_startup() -> None:
    checkpoint_path = Path(os.getenv("MODEL_PATH", "artifacts/improved_resnet18.pt"))
    device = select_device(os.getenv("DEVICE", "auto"))

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train a model first with train.py or set MODEL_PATH."
        )

    model, meta = load_checkpoint(checkpoint_path, device=device)
    app.state.model = model
    app.state.meta = meta
    app.state.checkpoint = str(checkpoint_path.resolve())

    # Initialize the clinical analyzer
    analyzer = ClinicalAnalyzer(
        model=model,
        model_name=str(meta["model_name"]),
        class_names=list(meta["class_names"]),
        image_size=int(meta["image_size"]),
        device=str(meta["device"]),
    )

    # Load calibration if available
    calibration_path = checkpoint_path.with_name(checkpoint_path.stem + "_calibration.json")
    if calibration_path.exists():
        analyzer.load_calibration(calibration_path)
        print(f"Loaded calibration from {calibration_path}")
    else:
        print(
            f"No calibration file found at {calibration_path}. "
            "Run calibrate.py for calibrated confidence and clinical metrics."
        )

    app.state.analyzer = analyzer


def _read_upload_as_pil(contents: bytes) -> Image.Image:
    """Parse uploaded bytes into a PIL Image."""
    try:
        return Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "checkpoint": app.state.checkpoint,
        "model_name": str(app.state.meta["model_name"]),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    """Original binary prediction endpoint (kept for backward compatibility)."""
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image = _read_upload_as_pil(contents)
    meta = app.state.meta

    prediction = predict_image(
        model=app.state.model,
        image=image,
        image_size=int(meta["image_size"]),
        device=str(meta["device"]),
        class_names=list(meta["class_names"]),
    )

    overlay, heatmap = generate_gradcam_overlay(
        model=app.state.model,
        model_name=str(meta["model_name"]),
        image=image,
        image_size=int(meta["image_size"]),
        device=str(meta["device"]),
        class_index=prediction["predicted_index"],
        alpha=0.45,
    )

    return {
        "prediction": prediction["prediction"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "heatmap_overlay_png_base64": encode_pil_to_base64(overlay),
        "heatmap_raw_png_base64": encode_pil_to_base64(heatmap_to_pil(heatmap)),
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    patient_id: str = Form("ANONYMOUS"),
) -> dict[str, object]:
    """Full clinical analysis endpoint.

    Returns severity scoring, calibrated confidence, uncertainty estimation,
    region analysis, Grad-CAM overlays, and a structured radiology-style report.
    """
    if not hasattr(app.state, "analyzer"):
        raise HTTPException(status_code=503, detail="Clinical analyzer not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image = _read_upload_as_pil(contents)
    analyzer: ClinicalAnalyzer = app.state.analyzer

    # Run full clinical analysis
    analysis = analyzer.analyze(image)

    # Generate structured text report
    report_text = generate_report(analysis, patient_id=patient_id)

    # Build JSON response
    result = analysis_to_dict(analysis)
    result["report"] = report_text

    return result
