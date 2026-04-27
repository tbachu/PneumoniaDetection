from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from modeling import TrainConfig, fit_model, resolve_data_root, save_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pneumonia classifier (baseline or improved).")
    parser.add_argument(
        "--stage",
        choices=["baseline", "improved"],
        default="baseline",
        help="baseline: simple CNN without augmentation, improved: stronger model + augmentation + regularization",
    )
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "resnet18", "efficientnet_b0"],
        default=None,
        help="Model backbone. If omitted, stage defaults are used.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/chest-xray-pneumonia/chest_xray",
        help="Path that contains train/val/test folders (or a parent folder containing them).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Checkpoint path (.pt)")

    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Run a small grid search over lr and weight decay before final training.",
    )
    parser.add_argument("--tune-epochs", type=int, default=2, help="Epochs per tuning trial.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run clinical evaluation (calibration + metrics + plots) after training.",
    )
    return parser


def defaults_for_stage(stage: str) -> dict[str, float | int | bool | str]:
    if stage == "baseline":
        return {
            "model_name": "simple_cnn",
            "epochs": 6,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.2,
            "augment": False,
            "pretrained": False,
            "use_scheduler": False,
            "label_smoothing": 0.0,
        }

    return {
        "model_name": "resnet18",
        "epochs": 10,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.35,
        "augment": True,
        "pretrained": True,
        "use_scheduler": True,
        "label_smoothing": 0.05,
    }


def choose_hparams_with_tuning(config: TrainConfig, tune_epochs: int) -> tuple[float, float]:
    search_space = [
        (1e-3, 1e-4),
        (3e-4, 1e-4),
        (1e-4, 1e-4),
        (3e-4, 1e-3),
    ]
    best_score = -1.0
    best_pair = (config.lr, config.weight_decay)

    print("Starting lightweight tuning sweep...")
    for lr, weight_decay in search_space:
        trial_cfg = replace(config, lr=lr, weight_decay=weight_decay, epochs=tune_epochs)
        print(f"Trial lr={lr} weight_decay={weight_decay}")
        result = fit_model(trial_cfg)
        score = float(result["best_val_acc"])
        print(f"Trial result val_acc={score:.4f}")

        if score > best_score:
            best_score = score
            best_pair = (lr, weight_decay)

    print(
        "Tuning complete. "
        f"Best lr={best_pair[0]} weight_decay={best_pair[1]} val_acc={best_score:.4f}"
    )
    return best_pair


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    stage_defaults = defaults_for_stage(args.stage)
    model_name = args.model or str(stage_defaults["model_name"])

    config = TrainConfig(
        data_root=str(resolve_data_root(args.data_dir)),
        model_name=model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs if args.epochs is not None else int(stage_defaults["epochs"]),
        lr=args.lr if args.lr is not None else float(stage_defaults["lr"]),
        weight_decay=(
            args.weight_decay if args.weight_decay is not None else float(stage_defaults["weight_decay"])
        ),
        augment=bool(stage_defaults["augment"]),
        pretrained=bool(stage_defaults["pretrained"]),
        use_scheduler=bool(stage_defaults["use_scheduler"]),
        label_smoothing=float(stage_defaults["label_smoothing"]),
        dropout=args.dropout if args.dropout is not None else float(stage_defaults["dropout"]),
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )

    if args.auto_tune:
        tuned_lr, tuned_weight_decay = choose_hparams_with_tuning(config, tune_epochs=args.tune_epochs)
        config = replace(config, lr=tuned_lr, weight_decay=tuned_weight_decay)

    print("Final training config:")
    print(config)
    result = fit_model(config)

    default_output = Path("artifacts") / f"{args.stage}_{config.model_name}.pt"
    output_path = Path(args.output) if args.output else default_output
    save_checkpoint(
        checkpoint_path=output_path,
        model=result["model"],
        model_name=config.model_name,
        class_names=result["class_names"],
        image_size=config.image_size,
        dropout=config.dropout,
        metadata={
            "stage": args.stage,
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
            "test_acc": result["test_acc"],
            "train_config": result["config"],
        },
    )

    print(f"Saved checkpoint: {output_path.resolve()}")

    # --- Optional: auto-run calibration + clinical evaluation --------------
    if args.eval:
        print("\n" + "=" * 60)
        print("Running post-training calibration and clinical evaluation...")
        print("=" * 60 + "\n")

        from calibrate import _collect_predictions, _compute_clinical_metrics
        from clinical_engine import ClinicalAnalyzer
        from clinical_eval import run_evaluation

        # Calibrate temperature
        analyzer = ClinicalAnalyzer(
            model=result["model"],
            model_name=config.model_name,
            class_names=result["class_names"],
            image_size=config.image_size,
            device=result["device"],
        )

        loaders, _ = make_dataloaders(
            data_root=config.data_root,
            image_size=config.image_size,
            batch_size=config.batch_size,
            augment=False,
            num_workers=config.num_workers,
        )
        temperature = analyzer.calibrate(loaders["val"])

        pneumonia_index = (
            result["class_names"].index("PNEUMONIA")
            if "PNEUMONIA" in result["class_names"]
            else 1
        )
        labels, preds, probs = _collect_predictions(
            result["model"], loaders["test"], result["device"], pneumonia_index,
        )
        metrics = _compute_clinical_metrics(labels, preds, probs, pneumonia_index)
        analyzer.clinical_metrics = metrics
        analyzer.save_calibration(
            output_path.with_name(output_path.stem + "_calibration.json")
        )
        print(f"Temperature: {temperature:.4f}")

        # Full clinical evaluation with plots
        run_evaluation(
            checkpoint_path=output_path,
            data_dir=config.data_root,
            output_dir=str(output_path.parent / "clinical_eval"),
            device=config.device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )


if __name__ == "__main__":
    main()
