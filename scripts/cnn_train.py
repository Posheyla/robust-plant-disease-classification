import os
import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset import PlantVillageDataset


# =========================
# Configuration
# =========================
SPLIT_CSV = "data/split.csv"
BASE_DIR = "raw_data/PlantVillage-Dataset"
OUTPUT_DIR = Path("results/custom_cnn")

BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMAGE_SIZE = 128
DROPOUT = 0.5
RANDOM_SEED = 42
USE_CLASS_WEIGHTS = False  # Set True for class-weighted CE
SAVE_BEST_ONLY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(RANDOM_SEED)


# =========================
# Transforms
# =========================
# For a custom CNN, ImageNet normalization is not strictly required.
# A simple [0,1] tensor pipeline is fine as a baseline.
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# =========================
# Model
# =========================
class PlantDiseaseCNN(nn.Module):
    """
    Baseline custom CNN aligned with the proposal:
    four convolutional blocks with increasing filters,
    ReLU activations, max pooling, and a FC head with dropout.
    """

    def __init__(self, num_classes: int, input_size: int = 128, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Infer flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            feat = self.features(dummy)
            flattened_dim = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# Utilities
# =========================
def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_class_weights(dataset: PlantVillageDataset, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for label_str in dataset.df["label"]:
        idx = dataset.label_to_idx[label_str]
        counts[idx] += 1.0

    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_shape": list(cm.shape),
    }, cm


def run_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# Main
# =========================
def main():
    ensure_output_dir(OUTPUT_DIR)

    # Shared label mapping from the unified split file
    label_to_idx = PlantVillageDataset.build_label_mapping(SPLIT_CSV)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    print(f"Number of classes: {num_classes}")
    print(f"Using device: {DEVICE}")

    # Datasets
    train_dataset = PlantVillageDataset(
        split_csv=SPLIT_CSV,
        split="train",
        data_type="color",
        transform=train_transform,
        base_dir=BASE_DIR,
        label_to_idx=label_to_idx,
    )

    val_dataset = PlantVillageDataset(
        split_csv=SPLIT_CSV,
        split="val",
        data_type="color",
        transform=eval_transform,
        base_dir=BASE_DIR,
        label_to_idx=label_to_idx,
    )

    test_color_dataset = PlantVillageDataset(
        split_csv=SPLIT_CSV,
        split="test",
        data_type="color",
        transform=eval_transform,
        base_dir=BASE_DIR,
        label_to_idx=label_to_idx,
    )

    test_seg_dataset = PlantVillageDataset(
        split_csv=SPLIT_CSV,
        split="test",
        data_type="segmented",
        transform=eval_transform,
        base_dir=BASE_DIR,
        label_to_idx=label_to_idx,
    )

    # Dataloaders
    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_color_loader = DataLoader(
        test_color_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_seg_loader = DataLoader(
        test_seg_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # Model
    model = PlantDiseaseCNN(
        num_classes=num_classes,
        input_size=IMAGE_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Loss
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_dataset, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class-weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history: List[Dict] = []
    best_val_acc = -1.0
    best_model_path = OUTPUT_DIR / "best_custom_cnn.pt"

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=DEVICE,
        )

        scheduler.step()

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to: {best_model_path}")

    # Save history
    save_json({"history": history}, OUTPUT_DIR / "training_history.json")

    # Load best model before final evaluation
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Loaded best model from: {best_model_path}")

    # Final evaluation
    print("\nEvaluating on test sets...")

    test_color_loss, test_color_acc = run_one_epoch(
        model=model,
        dataloader=test_color_loader,
        criterion=criterion,
        optimizer=None,
        device=DEVICE,
    )

    test_seg_loss, test_seg_acc = run_one_epoch(
        model=model,
        dataloader=test_seg_loader,
        criterion=criterion,
        optimizer=None,
        device=DEVICE,
    )

    y_true_color, y_pred_color, _ = get_predictions(model, test_color_loader, DEVICE)
    y_true_seg, y_pred_seg, _ = get_predictions(model, test_seg_loader, DEVICE)

    color_metrics, color_cm = evaluate_metrics(y_true_color, y_pred_color)
    seg_metrics, seg_cm = evaluate_metrics(y_true_seg, y_pred_seg)

    print("\nFinal Test Results:")
    print(f"Color Test    | Loss: {test_color_loss:.4f}, Acc: {test_color_acc:.4f}, Macro-F1: {color_metrics['macro_f1']:.4f}")
    print(f"Segmented Test| Loss: {test_seg_loss:.4f}, Acc: {test_seg_acc:.4f}, Macro-F1: {seg_metrics['macro_f1']:.4f}")

    # Save reports
    results = {
        "config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "image_size": IMAGE_SIZE,
            "dropout": DROPOUT,
            "device": str(DEVICE),
            "use_class_weights": USE_CLASS_WEIGHTS,
        },
        "test_color": {
            "loss": float(test_color_loss),
            "acc": float(test_color_acc),
            **color_metrics,
        },
        "test_segmented": {
            "loss": float(test_seg_loss),
            "acc": float(test_seg_acc),
            **seg_metrics,
        },
    }

    save_json(results, OUTPUT_DIR / "metrics.json")
    np.save(OUTPUT_DIR / "confusion_matrix_color.npy", color_cm)
    np.save(OUTPUT_DIR / "confusion_matrix_segmented.npy", seg_cm)

    # Optional text reports
    color_report = classification_report(
        y_true_color,
        y_pred_color,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        zero_division=0,
    )
    seg_report = classification_report(
        y_true_seg,
        y_pred_seg,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        zero_division=0,
    )

    with open(OUTPUT_DIR / "classification_report_color.txt", "w") as f:
        f.write(color_report)

    with open(OUTPUT_DIR / "classification_report_segmented.txt", "w") as f:
        f.write(seg_report)

    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()