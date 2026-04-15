from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from modeling import (
    generate_gradcam_overlay,
    heatmap_to_pil,
    load_checkpoint,
    predict_image,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for a chest X-ray image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument(
        "--output",
        default="artifacts/gradcam_overlay.png",
        help="Path to save the overlay heatmap image",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay strength")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    model, meta = load_checkpoint(args.checkpoint, device=device)
    image = Image.open(args.image).convert("RGB")

    prediction = predict_image(
        model=model,
        image=image,
        image_size=int(meta["image_size"]),
        device=meta["device"],
        class_names=list(meta["class_names"]),
    )

    overlay, heatmap = generate_gradcam_overlay(
        model=model,
        model_name=str(meta["model_name"]),
        image=image,
        image_size=int(meta["image_size"]),
        device=meta["device"],
        class_index=prediction["predicted_index"],
        alpha=args.alpha,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)

    raw_map_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}")
    heatmap_to_pil(heatmap).save(raw_map_path)

    print(f"Prediction: {prediction['prediction']} (confidence={prediction['confidence']:.4f})")
    print(f"Overlay saved to: {output_path.resolve()}")
    print(f"Raw heatmap saved to: {raw_map_path.resolve()}")


if __name__ == "__main__":
    main()
