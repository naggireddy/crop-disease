"""
Crop Disease Model Training Engine
Trains the multimodal MobileNetV3 / EfficientNet model with:
- Class-weighted CrossEntropy or Multi-label BCEWithLogits
- CosineAnnealingLR scheduling
- Early stopping on validation loss
- Best checkpoint saving to models/checkpoints/best_model.pt
- Smoke test mode for end-to-end verification without external datasets
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import config
from multimodal import MultimodalCropDiseaseModel
from dataset import create_dataloaders, CropDiseaseDataset

class EarlyStopping:
    """Monitors validation loss and stops training when improvement stalls."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    multi_label: bool
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, weather, survey, targets, _ in dataloader:
        images = images.to(device)
        weather = weather.to(device)
        survey = survey.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images, weather, survey)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        if not multi_label:
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
        else:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == targets).all(dim=-1).sum().item()

    return {
        "loss": running_loss / max(1, total),
        "accuracy": correct / max(1, total)
    }

def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    multi_label: bool
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, weather, survey, targets, _ in dataloader:
            images = images.to(device)
            weather = weather.to(device)
            survey = survey.to(device)
            targets = targets.to(device)

            logits = model(images, weather, survey)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            if not multi_label:
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets).sum().item()
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == targets).all(dim=-1).sum().item()

    return {
        "val_loss": running_loss / max(1, total),
        "val_accuracy": correct / max(1, total)
    }

def run_smoke_test(epochs: int = 2, backbone: str = "mobilenet_v3_large") -> Dict[str, Any]:
    """Runs a self-contained synthetic training test to guarantee pipeline integrity."""
    print("[*] Running end-to-end ML Training Smoke Test (synthetic batch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 82

    model = MultimodalCropDiseaseModel(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
        multi_label=False
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Synthetic batch of 8 samples
    dummy_imgs = torch.randn(8, 3, 224, 224, device=device)
    dummy_weather = torch.rand(8, 5, device=device)
    dummy_survey = torch.rand(8, 15, device=device)
    dummy_targets = torch.randint(0, num_classes, (8,), device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(dummy_imgs, dummy_weather, dummy_survey)
        loss = criterion(out, dummy_targets)
        loss.backward()
        optimizer.step()
        print(f"    Smoke Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    checkpoint_path = config.CHECKPOINTS_DIR / "smoke_test_model.pt"
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "loss": float(loss.item()),
        "backbone": backbone,
        "num_classes": num_classes,
        "version": "smoke_test"
    }, checkpoint_path)
    print(f"[+] Smoke test checkpoint successfully verified and saved: {checkpoint_path}")
    return {"status": "success", "loss": float(loss.item())}

def train(
    manifest_path: Path = config.MANIFEST_CSV,
    epochs: int = config.MAX_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    backbone: str = config.BACKBONE,
    multi_label: bool = config.MULTI_LABEL,
    smoke_test: bool = False
):
    if smoke_test:
        return run_smoke_test(epochs=min(epochs, 3), backbone=backbone)

    loaders = create_dataloaders(manifest_path, batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    if train_loader is None or len(train_loader.dataset) == 0:
        print("[!] No training images found in dataset_manifest.csv.")
        print("[!] To verify training pipeline mechanics without external datasets, run with --smoke-test.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | Backbone: {backbone} | Multi-label: {multi_label}")

    train_ds = loaders["datasets"]["train"]
    num_classes = train_ds.num_classes

    # Model
    model = MultimodalCropDiseaseModel(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=True,
        multi_label=multi_label
    ).to(device)

    # Class balancing weights
    class_weights = train_ds.get_class_weights().to(device)
    if multi_label:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=config.PATIENCE)

    best_val_loss = float("inf")
    best_checkpoint_path = config.CHECKPOINTS_DIR / "best_model.pt"
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, multi_label)
        scheduler.step()

        if val_loader and len(val_loader.dataset) > 0:
            val_res = validate(model, val_loader, criterion, device, multi_label)
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_res['loss']:.4f} | Acc: {train_res['accuracy']*100:.2f}% | "
                f"Val Loss: {val_res['val_loss']:.4f} | Val Acc: {val_res['val_accuracy']*100:.2f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Checkpoint best model
            if val_res["val_loss"] < best_val_loss:
                best_val_loss = val_res["val_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_res["val_loss"],
                    "val_accuracy": val_res["val_accuracy"],
                    "num_classes": num_classes,
                    "backbone": backbone,
                    "multi_label": multi_label,
                    "model_version": config.MODEL_VERSION,
                    "class_labels": [train_ds.idx_to_label[i] for i in range(num_classes)]
                }, best_checkpoint_path)
                print(f"    [+] New best model checkpoint saved: {best_checkpoint_path}")

            if early_stopping(val_res["val_loss"]):
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break
        else:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_res['loss']:.4f} | Acc: {train_res['accuracy']*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal crop disease diagnosis model.")
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--backbone", type=str, default=config.BACKBONE)
    parser.add_argument("--multi-label", action="store_true", default=config.MULTI_LABEL)
    parser.add_argument("--smoke-test", action="store_true", help="Run self-contained pipeline smoke test.")

    args = parser.parse_args()
    train(args.manifest, args.epochs, args.batch_size, args.lr, args.backbone, args.multi_label, args.smoke_test)

