# %% [markdown]
# # ChestAI Multi-Label Training — NIH ChestX-ray14
#
# This notebook trains a ResNet-18 model to detect **14 chest pathologies**
# simultaneously from frontal chest X-rays using the NIH ChestX-ray14 dataset.
#
# **Output:** A `.pt` checkpoint compatible with the ChestAI dashboard.
#
# ### Setup
# 1. Create a new Kaggle Notebook
# 2. Add the dataset: **nih-chest-xrays/data**
# 3. Enable GPU: Settings → Accelerator → GPU T4 ×2 (or P100)
# 4. Paste this script and run

# %%
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore", category=UserWarning)

# %%
# ─── CONFIG ───────────────────────────────────────────────────────────────
BATCH_SIZE = 64
IMAGE_SIZE = 224
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.35
NUM_WORKERS = 2
SEED = 42

FINDING_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(FINDING_LABELS)

# Automatically detect the dataset location on Kaggle
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_OUTPUT = Path("/kaggle/working")

# The NIH dataset may be at different paths depending on how it's added
_POSSIBLE_ROOTS = [
    KAGGLE_INPUT / "datasets" / "organizations" / "nih-chest-xrays" / "data",
    KAGGLE_INPUT / "data",
    KAGGLE_INPUT / "nih-chest-xrays" / "data",
    KAGGLE_INPUT / "nih-chest-xrays",
]

DATA_ROOT = None
for _r in _POSSIBLE_ROOTS:
    if _r.is_dir():
        DATA_ROOT = _r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find NIH dataset. Tried: {[str(p) for p in _POSSIBLE_ROOTS]}. "
        "Make sure you've added the 'nih-chest-xrays/data' dataset to this notebook."
    )

print(f"Dataset root: {DATA_ROOT}")
print(f"Contents: {[p.name for p in DATA_ROOT.iterdir()]}")

# %%
# ─── SEED ─────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# %%
# ─── LOAD LABELS ──────────────────────────────────────────────────────────

# Find the CSV file
csv_candidates = list(DATA_ROOT.glob("Data_Entry*.csv"))
if not csv_candidates:
    csv_candidates = list(DATA_ROOT.glob("**/Data_Entry*.csv"))
if not csv_candidates:
    raise FileNotFoundError("Could not find Data_Entry CSV file in the dataset.")

CSV_PATH = csv_candidates[0]
print(f"Labels CSV: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Total images in CSV: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))

# %%
# ─── BUILD IMAGE PATH INDEX ──────────────────────────────────────────────

# Find all image files (they may be in subdirectories like images_001/images/)
print("Scanning for images...")
all_images = {}
for img_path in DATA_ROOT.rglob("*.png"):
    all_images[img_path.name] = str(img_path)

print(f"Found {len(all_images)} images on disk")

# Filter CSV to only images we actually have
df = df[df["Image Index"].isin(all_images)]
df["image_path"] = df["Image Index"].map(all_images)
print(f"Matched images: {len(df)}")

# %%
# ─── MULTI-HOT ENCODING ──────────────────────────────────────────────────

def encode_labels(finding_labels_str: str) -> np.ndarray:
    """Convert pipe-separated labels to a multi-hot vector."""
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)
    findings = finding_labels_str.split("|")
    for f in findings:
        f = f.strip()
        if f in FINDING_LABELS:
            labels[FINDING_LABELS.index(f)] = 1.0
    return labels

df["labels"] = df["Finding Labels"].apply(encode_labels)

# Show label distribution
label_counts = np.stack(df["labels"].values).sum(axis=0)
print("\nLabel distribution:")
for name, count in sorted(zip(FINDING_LABELS, label_counts), key=lambda x: -x[1]):
    pct = count / len(df) * 100
    print(f"  {name:<22s} {int(count):>6d}  ({pct:.1f}%)")

no_finding = (np.stack(df["labels"].values).sum(axis=1) == 0).sum()
print(f"\n  No Finding:            {no_finding:>6d}  ({no_finding/len(df)*100:.1f}%)")

# %%
# ─── PATIENT-LEVEL SPLIT ─────────────────────────────────────────────────
# Split by Patient ID to prevent data leakage

# Extract patient ID from image name (format: XXXXX_YYY.png)
df["patient_id"] = df["Image Index"].str.split("_").str[0].astype(int)

# 80% train, 10% val, 10% test — grouped by patient
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, temp_idx = next(gss1.split(df, groups=df["patient_id"]))

df_train = df.iloc[train_idx].reset_index(drop=True)
df_temp = df.iloc[temp_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["patient_id"]))

df_val = df_temp.iloc[val_idx].reset_index(drop=True)
df_test = df_temp.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
print(f"Train patients: {df_train['patient_id'].nunique()}")
print(f"Val patients: {df_val['patient_id'].nunique()}")
print(f"Test patients: {df_test['patient_id'].nunique()}")

# Sanity check: no patient overlap
assert set(df_train["patient_id"]) & set(df_val["patient_id"]) == set()
assert set(df_train["patient_id"]) & set(df_test["patient_id"]) == set()
print("✓ No patient overlap between splits")

# %%
# ─── DATASET CLASS ────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ChestXrayDataset(Dataset):
    """NIH ChestX-ray14 dataset with multi-hot labels."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row["labels"], dtype=torch.float32)
        return img, labels


train_dataset = ChestXrayDataset(df_train, transform=train_transform)
val_dataset = ChestXrayDataset(df_val, transform=eval_transform)
test_dataset = ChestXrayDataset(df_test, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

# %%
# ─── CLASS WEIGHTS ────────────────────────────────────────────────────────
# Handle class imbalance with positive weights for BCE loss

label_matrix = np.stack(df_train["labels"].values)
pos_counts = label_matrix.sum(axis=0)
neg_counts = len(label_matrix) - pos_counts
pos_weights = neg_counts / (pos_counts + 1e-5)
pos_weights = np.clip(pos_weights, 1.0, 20.0)  # cap to avoid extreme weights

pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)
print("\nPositive weights (neg/pos ratio):")
for name, w in zip(FINDING_LABELS, pos_weights):
    print(f"  {name:<22s} {w:.1f}")

# %%
# ─── MODEL ────────────────────────────────────────────────────────────────

def create_multilabel_model(num_classes=14, dropout=0.35, pretrained=True):
    """ResNet-18 with multi-label sigmoid head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
        # No sigmoid here — BCEWithLogitsLoss applies it internally
    )
    return model


model = create_multilabel_model(NUM_CLASSES, DROPOUT).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# ─── TRAINING ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}", flush=True)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    probs = torch.sigmoid(logits_cat).numpy()
    labels_np = labels_cat.numpy()

    # Per-class AUC
    aucs = []
    for i in range(NUM_CLASSES):
        if labels_np[:, i].sum() > 0 and labels_np[:, i].sum() < len(labels_np):
            auc_i = roc_auc_score(labels_np[:, i], probs[:, i])
            aucs.append(auc_i)
        else:
            aucs.append(float("nan"))

    mean_auc = np.nanmean(aucs)
    return avg_loss, mean_auc, aucs


# %%
# Training loop
print("=" * 60)
print("Starting training...")
print("=" * 60)

history = []
best_val_auc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_auc, val_aucs = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_auc": val_auc,
    })

    # Print per-class AUC every few epochs
    print(f"\nEpoch {epoch}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | "
          f"val_mean_AUC={val_auc:.4f}")

    if epoch == 1 or epoch == EPOCHS or epoch % 2 == 0:
        for name, auc_val in zip(FINDING_LABELS, val_aucs):
            print(f"  {name:<22s} AUC={auc_val:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ★ New best model (mean AUC = {val_auc:.4f})")

print(f"\nBest validation mean AUC: {best_val_auc:.4f}")

# %%
# ─── EVALUATE ON TEST SET ────────────────────────────────────────────────

# Load best model
model.load_state_dict(best_state)
model = model.to(DEVICE)

test_loss, test_auc, test_aucs = evaluate(model, test_loader, criterion, DEVICE)

print("=" * 60)
print(f"TEST SET RESULTS — Mean AUC: {test_auc:.4f}")
print("=" * 60)
for name, auc_val in zip(FINDING_LABELS, test_aucs):
    bar = "█" * int(auc_val * 20) if not np.isnan(auc_val) else "?"
    print(f"  {name:<22s} AUC={auc_val:.4f}  {bar}")

# %%
# ─── SAVE CHECKPOINT ─────────────────────────────────────────────────────
# Format compatible with ChestAI's load_checkpoint()

output_path = KAGGLE_OUTPUT / "multilabel_resnet18.pt"

checkpoint = {
    "model_name": "resnet18",
    "class_names": FINDING_LABELS,
    "image_size": IMAGE_SIZE,
    "dropout": DROPOUT,
    "state_dict": best_state,
    "metadata": {
        "task": "multilabel",
        "stage": "multilabel_nih14",
        "dataset": "NIH ChestX-ray14",
        "num_classes": NUM_CLASSES,
        "best_val_auc": float(best_val_auc),
        "test_mean_auc": float(test_auc),
        "test_per_class_auc": {name: float(a) for name, a in zip(FINDING_LABELS, test_aucs)},
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
    },
}

torch.save(checkpoint, output_path)
print(f"\n✓ Checkpoint saved: {output_path}")
print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# %%
# ─── TRAINING CURVES ─────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = [h["epoch"] for h in history]

ax1.plot(epochs_range, [h["train_loss"] for h in history], "o-", label="Train", color="#4F8FF7")
ax1.plot(epochs_range, [h["val_loss"] for h in history], "o-", label="Val", color="#F87171")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCE Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, [h["val_auc"] for h in history], "o-", color="#34D399", lw=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean AUC-ROC")
ax2.set_title("Validation Mean AUC")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(KAGGLE_OUTPUT / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ─── PER-CLASS AUC BAR CHART ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

sorted_pairs = sorted(zip(FINDING_LABELS, test_aucs), key=lambda x: x[1], reverse=True)
names = [p[0] for p in sorted_pairs]
values = [p[1] for p in sorted_pairs]
colors = ["#34D399" if v >= 0.8 else "#FBBF24" if v >= 0.7 else "#F87171" for v in values]

bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)
ax.set_xlabel("AUC-ROC")
ax.set_title(f"Per-Pathology AUC-ROC (Mean = {test_auc:.3f})")
ax.set_xlim(0.5, 1.0)
ax.axvline(x=0.8, color="gray", ls="--", alpha=0.4, label="Good (0.80)")
ax.legend()
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

for bar, val in zip(bars, values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10)

fig.tight_layout()
fig.savefig(KAGGLE_OUTPUT / "per_class_auc.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print("\n" + "=" * 60)
print("DONE! Download these files:")
print(f"  1. {output_path.name}  (model checkpoint)")
print("  2. training_curves.png")
print("  3. per_class_auc.png")
print("=" * 60)
print("\nTo use locally:")
print("  1. Copy multilabel_resnet18.pt to artifacts/")
print("  2. Run: MODEL_PATH=artifacts/multilabel_resnet18.pt uvicorn app:app")
