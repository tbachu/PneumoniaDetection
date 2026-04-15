# PneumoniaDetection

End-to-end pneumonia detection pipeline using chest X-rays.

This repo now includes:
- Baseline model training (simple CNN)
- Improved model training (ResNet18 + augmentation + regularization)
- Grad-CAM explainability heatmaps
- FastAPI service for prediction + heatmap return

## 1) Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Download Dataset

```bash
python data_loader.py
```

Expected split root after download/move:

```text
data/chest-xray-pneumonia/chest_xray/
	train/
	val/
	test/
```

## 3) Step 2 - Baseline Model

Train a simple CNN for `PNEUMONIA` vs `NORMAL`:

```bash
python train.py --stage baseline
```

Default checkpoint:

```text
artifacts/baseline_simple_cnn.pt
```

## 4) Step 3 - Improve It

Improved training uses:
- Data augmentation
- Better architecture (default: `resnet18`)
- Regularization and scheduler
- Optional hyperparameter tuning sweep

Run improved training:

```bash
python train.py --stage improved
```

Run improved training with tuning:

```bash
python train.py --stage improved --auto-tune
```

Use EfficientNet instead:

```bash
python train.py --stage improved --model efficientnet_b0
```

Default checkpoint:

```text
artifacts/improved_resnet18.pt
```

## 5) Step 4 - Explainability with Grad-CAM

Generate a Grad-CAM overlay for one image:

```bash
python gradcam.py \
	--checkpoint artifacts/improved_resnet18.pt \
	--image data/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg \
	--output artifacts/gradcam_overlay.png
```

This saves:
- Overlay image: `artifacts/gradcam_overlay.png`
- Raw heatmap: `artifacts/gradcam_overlay_raw.png`

## 6) Step 5 - FastAPI Endpoint

Start API:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predict with image upload:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-F "file=@data/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
```

Response includes:
- `prediction`
- `confidence`
- `probabilities`
- `heatmap_overlay_png_base64`
- `heatmap_raw_png_base64`

## Latest Run Results (2026-04-15)

Improved model run:
- Command: `python train.py --stage improved --epochs 10 --device mps --num-workers 0`
- Checkpoint: `artifacts/improved_resnet18.pt`
- Best validation accuracy: `0.9375`
- Test accuracy: `0.9311`

Generated Grad-CAM artifacts:
- `artifacts/gradcam_normal.png`
- `artifacts/gradcam_normal_raw.png`
- `artifacts/gradcam_pneumonia.png`
- `artifacts/gradcam_pneumonia_raw.png`

API validation summary:
- `/health`: `200 OK`
- `/predict`: `200 OK`
- Example prediction on test PNEUMONIA image: `PNEUMONIA` (confidence `0.9795`)

## Notes

- If you trained a different checkpoint, set it before starting API:

```bash
export MODEL_PATH=artifacts/baseline_simple_cnn.pt
uvicorn app:app --reload
```

- Device override:

```bash
export DEVICE=cpu
```