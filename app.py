from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from modeling import (
    encode_pil_to_base64,
    generate_gradcam_overlay,
    heatmap_to_pil,
    load_checkpoint,
    predict_image,
    select_device,
)

app = FastAPI(title="Pneumonia Detection API", version="1.0.0")


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
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc

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
