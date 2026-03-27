import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from typing import Optional

from dataset import PlantVillageDataset


# =========================
# Configuration
# =========================
SPLIT_CSV = "data/split.csv"
BASE_DIR = "raw_data/PlantVillage-Dataset"

BATCH_SIZE = 32
NUM_EPOCHS = 1 # increase later with GPU
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # use 0 first for debugging; can increase later
IMAGE_SIZE = 128 # increase later

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# =========================
# Build shared label mapping
# =========================
label_to_idx = PlantVillageDataset.build_label_mapping(SPLIT_CSV)
num_classes = len(label_to_idx)

print(f"Number of classes: {num_classes}")
print(f"Using device: {DEVICE}")


# =========================
# Datasets
# =========================
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
    transform=val_transform,
    base_dir=BASE_DIR,
    label_to_idx=label_to_idx,
)

train_dataset.df = train_dataset.df.sample(5000, random_state=42).reset_index(drop=True)
val_dataset.df = val_dataset.df.sample(1000, random_state=42).reset_index(drop=True)

test_color_dataset = PlantVillageDataset(
    split_csv=SPLIT_CSV,
    split="test",
    data_type="color",
    transform=val_transform,
    base_dir=BASE_DIR,
    label_to_idx=label_to_idx,
)

test_seg_dataset = PlantVillageDataset(
    split_csv=SPLIT_CSV,
    split="test",
    data_type="segmented",
    transform=val_transform,
    base_dir=BASE_DIR,
    label_to_idx=label_to_idx,
)


# =========================
# Dataloaders
# =========================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_color_loader = DataLoader(
    test_color_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_seg_loader = DataLoader(
    test_seg_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# =========================
# Model
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)


# =========================
# Loss / Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================
# Helper functions
# =========================
def run_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

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
# Training loop
# =========================
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

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )


# =========================
# Final evaluation
# =========================
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

print("\nFinal Test Results:")
print(f"Color Test    | Loss: {test_color_loss:.4f}, Acc: {test_color_acc:.4f}")
print(f"Segmented Test| Loss: {test_seg_loss:.4f}, Acc: {test_seg_acc:.4f}")