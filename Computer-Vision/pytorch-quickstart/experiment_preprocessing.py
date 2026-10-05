"""
Does CenterCrop hurt? A controlled preprocessing comparison.

analyze_errors.py found that Resize(256) + CenterCrop(224) discards ~49% of
each source image. That is alarming on its face, but discarding pixels only
matters if the discarded pixels carried disease evidence. This measures it
instead of assuming.

  Variant A: Resize(256) -> CenterCrop(224)   [current pipeline]
  Variant B: Resize((224, 224))               [whole image, aspect squashed]

Everything downstream is held identical: same frozen backbone, same head
architecture, same seed, same epochs.
"""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

DATA_ROOT = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/original")
SPLITS = {
    "train": DATA_ROOT / "Train" / "Train",
    "val": DATA_ROOT / "Validation" / "Validation",
    "test": DATA_ROOT / "Test" / "Test",
}

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

VARIANTS = {
    "A: Resize(256)+CenterCrop(224)": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE,
    ]),
    "B: Resize((224,224)) full image": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NORMALIZE,
    ]),
}

HEAD_EPOCHS = 40
HEAD_BATCH = 64


def build_backbone():
    backbone = models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    backbone.classifier = nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def extract(backbone, transform, path):
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    feats, labs = [], []
    for images, labels in loader:
        feats.append(backbone(images))
        labs.append(labels)
    return torch.cat(feats), torch.cat(labs), dataset.classes


def train_head(train_features, train_labels, num_classes, seed=0):
    """Deterministic head training — fixed seed, manual batching (no worker RNG)."""
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[1], num_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    n = len(train_labels)
    head.train()
    for _ in range(HEAD_EPOCHS):
        order = torch.randperm(n, generator=generator)
        for i in range(0, n, HEAD_BATCH):
            idx = order[i : i + HEAD_BATCH]
            loss = loss_fn(head(train_features[idx]), train_labels[idx])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return head


def accuracy(head, features, labels):
    head.eval()
    with torch.no_grad():
        return (head(features).argmax(1) == labels).float().mean().item()


backbone = build_backbone()
results = {}

for name, transform in VARIANTS.items():
    print(f"\n{name}")
    print("-" * 60)
    cached = {}
    start = time.perf_counter()
    for split, path in SPLITS.items():
        feats, labs, classes = extract(backbone, transform, path)
        cached[split] = (feats, labs)
        print(f"  {split:<6} {len(labs):>5} images")
    extract_time = time.perf_counter() - start

    head = train_head(*cached["train"], num_classes=len(classes))
    val_acc = accuracy(head, *cached["val"])
    test_acc = accuracy(head, *cached["test"])
    results[name] = (val_acc, test_acc, extract_time)
    print(f"  extraction {extract_time:.0f}s | val {val_acc*100:.1f}% | test {test_acc*100:.1f}%")

print("\n" + "=" * 72)
print("COMPARISON")
print("=" * 72)
print(f"{'Variant':<34} {'Val':>8} {'Test':>8} {'Extract':>10}")
print("-" * 72)
for name, (val_acc, test_acc, extract_time) in results.items():
    print(f"{name:<34} {val_acc*100:>7.1f}% {test_acc*100:>7.1f}% {extract_time:>9.0f}s")

test_scores = [r[1] for r in results.values()]
delta = (test_scores[1] - test_scores[0]) * 100
print(f"\nDifference (B - A) on test: {delta:+.1f} percentage points")
print(f"On a 150-image test set, 1 image = {100/150:.1f} percentage points.")
