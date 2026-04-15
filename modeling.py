from __future__ import annotations

import base64
import copy
import random
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASS_NAMES_DEFAULT = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root).resolve()

    def has_expected_splits(path: Path) -> bool:
        return all((path / split).is_dir() for split in ("train", "val", "test"))

    if has_expected_splits(root):
        return root

    for subdir in root.rglob("*"):
        if subdir.is_dir() and has_expected_splits(subdir):
            return subdir

    raise FileNotFoundError(
        f"Could not find train/val/test folders under {root}. "
        "Pass --data-dir pointing at the chest_xray split root."
    )


def build_transforms(image_size: int, augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def make_dataloaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    augment: bool,
    num_workers: int = 2,
) -> tuple[dict[str, DataLoader], list[str]]:
    resolved_root = resolve_data_root(data_root)
    train_transform, eval_transform = build_transforms(image_size=image_size, augment=augment)

    train_dataset = datasets.ImageFolder(resolved_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(resolved_root / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(resolved_root / "test", transform=eval_transform)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, train_dataset.classes


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "simple_cnn":
        return model.features[-3]
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def disable_inplace_relu(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


@dataclass
class TrainConfig:
    data_root: str
    model_name: str
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    augment: bool = False
    pretrained: bool = False
    use_scheduler: bool = False
    label_smoothing: float = 0.0
    dropout: float = 0.3
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += labels.size(0)

    return running_loss / running_total, running_correct / running_total


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

    return running_loss / running_total, running_correct / running_total


def fit_model(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)

    loaders, class_names = make_dataloaders(
        data_root=config.data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        augment=config.augment,
        num_workers=config.num_workers,
    )

    model = create_model(
        model_name=config.model_name,
        num_classes=len(class_names),
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.use_scheduler
        else None
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = _eval_epoch(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_loss, test_acc = _eval_epoch(model, loaders["test"], criterion, device)
    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch} | test_acc={test_acc:.4f}", flush=True)

    return {
        "model": model,
        "class_names": class_names,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "device": device,
        "config": asdict(config),
    }


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    model_name: str,
    class_names: list[str],
    image_size: int,
    dropout: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "class_names": class_names,
        "image_size": image_size,
        "dropout": dropout,
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(checkpoint_path: str | Path, device: str = "auto") -> tuple[nn.Module, dict[str, Any]]:
    resolved_device = select_device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=resolved_device)

    model_name = checkpoint["model_name"]
    class_names = checkpoint.get("class_names", CLASS_NAMES_DEFAULT)
    image_size = int(checkpoint.get("image_size", 224))
    dropout = float(checkpoint.get("dropout", 0.3))

    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
        dropout=dropout,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metadata = {
        "model_name": model_name,
        "class_names": class_names,
        "image_size": image_size,
        "dropout": dropout,
        "device": resolved_device,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
    }
    return model, metadata


def preprocess_pil_image(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_image(
    model: nn.Module,
    image: Image.Image,
    image_size: int,
    device: str,
    class_names: list[str],
) -> dict[str, Any]:
    input_tensor = preprocess_pil_image(image, image_size=image_size).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(probs))
    probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return {
        "predicted_index": predicted_index,
        "prediction": class_names[predicted_index],
        "confidence": float(probs[predicted_index]),
        "probabilities": probabilities,
    }


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[:, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_map = cam.squeeze().detach().cpu().numpy()
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        return cam_map, logits.detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def heatmap_to_pil(heatmap: np.ndarray) -> Image.Image:
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(heatmap_uint8, mode="L")


def blend_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    resized = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    base_array = np.array(resized, dtype=np.float32)
    color_map = (cm.get_cmap("jet")(heatmap)[..., :3] * 255).astype(np.float32)
    blended = np.clip(alpha * color_map + (1 - alpha) * base_array, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def generate_gradcam_overlay(
    model: nn.Module,
    model_name: str,
    image: Image.Image,
    image_size: int,
    device: str,
    class_index: int | None = None,
    alpha: float = 0.45,
) -> tuple[Image.Image, np.ndarray]:
    model.eval()
    disable_inplace_relu(model)
    input_tensor = preprocess_pil_image(image, image_size=image_size).to(device)
    cam = GradCAM(model, target_layer=get_target_layer(model, model_name))
    try:
        heatmap, _ = cam.generate(input_tensor=input_tensor, class_index=class_index)
    finally:
        cam.close()

    overlay = blend_heatmap(image=image, heatmap=heatmap, alpha=alpha)
    return overlay, heatmap


def encode_pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
