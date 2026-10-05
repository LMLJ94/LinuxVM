"""
Plant disease classification via transfer learning (CPU-only).

Implements the strategy chosen in benchmark_cpu.py:

  1. ResNet18, ImageNet-pretrained, backbone FROZEN
  2. Push every image through the backbone ONCE, cache the 512-dim feature
     vectors to disk
  3. Train a small classifier head on the cached features

Why caching matters even more here than the benchmark suggested: the source
images are ~4000x3000 (12 megapixels). Decoding a JPEG that large costs more
than the network forward pass does. Caching pays that decode once instead of
once per epoch.

Dataset layout (torchvision ImageFolder infers labels from directory names):

    images/original/Train/Train/{Healthy,Powdery,Rust}/*.jpg
    images/original/Validation/Validation/{...}
    images/original/Test/Test/{...}
"""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

torch.manual_seed(0)

DATA_ROOT = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/original")
CACHE_DIR = Path("feature_cache")
BACKBONE_NAME = "resnet18"  # stamped into the cache so stale features fail loudly
SPLITS = {
    "train": DATA_ROOT / "Train" / "Train",
    "val": DATA_ROOT / "Validation" / "Validation",
    "test": DATA_ROOT / "Test" / "Test",
}

IMAGE_SIZE = 224
RESIZE_SHORT = 256
NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet channel statistics
NORM_STD = [0.229, 0.224, 0.225]
EXTRACT_BATCH = 32
NUM_WORKERS = 4  # parallel JPEG decode — the real bottleneck at 12MP
FEATURE_DIM = 512  # ResNet18 pooled output width (MobileNetV3-Small was 576)

HEAD_EPOCHS = 40
HEAD_BATCH = 64
HEAD_LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ---------------------------------------------------------------------------
# Preprocessing
#
# Pretrained models expect the same preprocessing used during their original
# ImageNet training: resize, center crop to 224, and normalize with ImageNet
# channel statistics. Skipping Normalize silently degrades accuracy, because
# the frozen filters were fitted to that input distribution.
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(RESIZE_SHORT),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def build_backbone():
    """Pretrained ResNet18 with its classifier removed.

    ResNet names its final layer .fc (MobileNet calls it .classifier).
    Replacing it with Identity makes forward() return the 512-dim pooled
    feature vector instead of 1000 ImageNet logits.
    """
    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone.to(device)


@torch.no_grad()
def extract_features(backbone, split_name, split_path):
    """Run one split through the frozen backbone, caching the result."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{split_name}.pt"

    if cache_file.exists():
        blob = torch.load(cache_file, weights_only=True)
        cached_backbone = blob.get("backbone", "unknown")
        if cached_backbone != BACKBONE_NAME:
            raise RuntimeError(
                f"{cache_file} holds {cached_backbone} features, but this "
                f"script uses {BACKBONE_NAME}. Delete {CACHE_DIR}/ and rerun."
            )
        print(f"  {split_name:<6} loaded {len(blob['labels']):>5} cached features")
        return blob["features"], blob["labels"], blob["classes"]

    dataset = datasets.ImageFolder(split_path, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=EXTRACT_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_features, all_labels = [], []
    start = time.perf_counter()
    for images, labels in loader:
        feats = backbone(images.to(device))
        all_features.append(feats.cpu())
        all_labels.append(labels)
    elapsed = time.perf_counter() - start

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    torch.save(
        {
            "features": features,
            "labels": labels,
            "classes": dataset.classes,
            "backbone": BACKBONE_NAME,
        },
        cache_file,
    )
    rate = len(labels) / elapsed
    print(
        f"  {split_name:<6} extracted {len(labels):>5} images in "
        f"{elapsed:>6.1f}s ({rate:.1f} img/s) -> {cache_file}"
    )
    return features, labels, dataset.classes


def evaluate(head, features, labels):
    head.eval()
    with torch.no_grad():
        preds = head(features.to(device)).argmax(1).cpu()
    return (preds == labels).float().mean().item(), preds


# ---------------------------------------------------------------------------
# 1. Feature extraction (the expensive step — paid once)
# ---------------------------------------------------------------------------
print("\nExtracting features with frozen ResNet18 backbone")
print("-" * 60)
backbone = build_backbone()

cached = {}
classes = None
for name, path in SPLITS.items():
    feats, labels, split_classes = extract_features(backbone, name, path)
    cached[name] = (feats, labels)
    classes = split_classes

print(f"\nClasses: {classes}")
print(f"Feature shape: {tuple(cached['train'][0].shape)}")

# ---------------------------------------------------------------------------
# 2. Train the classifier head on cached features (the cheap step)
# ---------------------------------------------------------------------------
train_features, train_labels = cached["train"]
val_features, val_labels = cached["val"]
test_features, test_labels = cached["test"]

head = nn.Linear(FEATURE_DIM, len(classes)).to(device)
optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
loss_fn = nn.CrossEntropyLoss()

train_loader = DataLoader(
    TensorDataset(train_features, train_labels),
    batch_size=HEAD_BATCH,
    shuffle=True,
)

print(f"\nTraining classifier head ({FEATURE_DIM} -> {len(classes)})")
print("-" * 60)

start = time.perf_counter()
for epoch in range(1, HEAD_EPOCHS + 1):
    head.train()
    epoch_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        loss = loss_fn(head(batch_features), batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        val_acc, _ = evaluate(head, val_features, val_labels)
        print(
            f"  epoch {epoch:>3}  loss {epoch_loss / len(train_loader):.4f}  "
            f"val acc {val_acc * 100:.1f}%"
        )
head_time = time.perf_counter() - start
print(f"\nHead training finished in {head_time:.1f}s")

# ---------------------------------------------------------------------------
# 3. Final evaluation
# ---------------------------------------------------------------------------
test_acc, test_preds = evaluate(head, test_features, test_labels)
val_acc, _ = evaluate(head, val_features, val_labels)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Validation accuracy: {val_acc * 100:.1f}%")
print(f"Test accuracy:       {test_acc * 100:.1f}%")

print("\nPer-class accuracy (test):")
for i, class_name in enumerate(classes):
    mask = test_labels == i
    class_acc = (test_preds[mask] == test_labels[mask]).float().mean().item()
    print(f"  {class_name:<10} {class_acc * 100:>5.1f}%  ({mask.sum().item()} images)")

print("\nConfusion matrix (rows = actual, cols = predicted):")
confusion = torch.zeros(len(classes), len(classes), dtype=torch.int32)
for actual, predicted in zip(test_labels, test_preds):
    confusion[actual, predicted] += 1
header = " " * 12 + "".join(f"{c[:8]:>10}" for c in classes)
print(header)
for i, class_name in enumerate(classes):
    row = "".join(f"{confusion[i, j].item():>10}" for j in range(len(classes)))
    print(f"  {class_name:<10}{row}")

# ---------------------------------------------------------------------------
# 4. Save the classifier head
#
# Only the head needs saving — the backbone is unmodified pretrained weights
# that torchvision can reproduce on demand.
#
# But the head is meaningless on its own: 1539 numbers that mean something only
# for 512-dim ResNet18 features built from images prepared exactly this way.
# Feed it MobileNet features, or skip Normalize, and it returns confident
# nonsense rather than an error. So record the input recipe beside the weights.
# The constants are read, not retyped, so the record cannot drift away from the
# transform actually used above.
# ---------------------------------------------------------------------------
torch.save(
    {
        "head": head.state_dict(),
        "classes": classes,
        "backbone": BACKBONE_NAME,
        "feature_dim": FEATURE_DIM,
        "preprocess": {
            "resize_short": RESIZE_SHORT,
            "center_crop": IMAGE_SIZE,
            "normalize_mean": NORM_MEAN,
            "normalize_std": NORM_STD,
        },
    },
    "plant_disease_head.pth",
)
print("\nSaved classifier head to plant_disease_head.pth")
