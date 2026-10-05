"""
Is ResNet18 really tied with MobileNetV3-Small, or did one seed get lucky?

Single-run results were 96.7% (ResNet18) vs 96.0% (MobileNetV3-Small) on 150
test images — a one-image gap. That is inside the noise. Rather than trust one
number, train the same linear head several times with different random seeds
and look at the spread.

The seed controls two things: the head's initial weights and the order the
training batches are shuffled in. Neither should matter if the result is real.

DINOv2-Small was added later as a third backbone: a self-supervised ViT whose
frozen features are known to be strong on fine texture. Run
extract_dinov2_features.py once first to create its cache.

YOLOv8n-cls and YOLOv8s-cls were added after that, as frozen backbones. Run
extract_yolo_features.py once first to create their caches.

Besides the accuracy summary, it prints which test images each backbone gets
wrong (in how many of the seeds), because tied totals can hide different errors.

Uses the cached feature vectors only, so each head takes ~1s to train.
"""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# Same recipe as plant_disease_transfer.py, so results are comparable
HEAD_EPOCHS = 40
HEAD_BATCH = 64
HEAD_LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]

CACHES = {
    "MobileNetV3-Small": Path("feature_cache_mobilenet_v3_small"),
    "ResNet18": Path("feature_cache"),
    # Created by extract_dinov2_features.py
    "DINOv2-Small": Path("feature_cache_dinov2_small"),
    # Created by extract_yolo_features.py
    "YOLOv8n-cls": Path("feature_cache_yolov8n_cls"),
    "YOLOv8s-cls": Path("feature_cache_yolov8s_cls"),
}
TEST_DIR = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/original/Test/Test")


def load_split(cache_dir, split):
    blob = torch.load(cache_dir / f"{split}.pt", weights_only=True)
    return blob["features"], blob["labels"], blob["classes"]


# Older caches did not store file names; they follow ImageFolder's sorted order.
# Newer caches do, and must agree with that order, or the per-image table would
# silently compare different images.
TEST_FILES = [Path(p).stem for p, _ in datasets.ImageFolder(TEST_DIR).samples]


def check_file_order(cache_dir):
    blob = torch.load(cache_dir / "test.pt", weights_only=True)
    if "files" in blob and blob["files"] != TEST_FILES:
        raise RuntimeError(f"{cache_dir}/test.pt lists its images in a different order")


def train_head(train_features, train_labels, num_classes, seed):
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[1], num_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=HEAD_BATCH,
        shuffle=True,
    )
    head.train()
    for _ in range(HEAD_EPOCHS):
        for feats, labels in loader:
            loss = loss_fn(head(feats), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return head


@torch.no_grad()
def accuracy(head, features, labels):
    head.eval()
    preds = head(features).argmax(1)
    return (preds == labels).float().mean().item()


results = {}
val_results = {}
wrong_counts = {}  # backbone -> per-test-image number of seeds it was wrong in
for name, cache_dir in CACHES.items():
    check_file_order(cache_dir)
    train_x, train_y, classes = load_split(cache_dir, "train")
    test_x, test_y, _ = load_split(cache_dir, "test")
    val_x, val_y, _ = load_split(cache_dir, "val")
    print(f"\n{name}  (feature dim {train_x.shape[1]})")
    print("-" * 40)

    accs = []
    val_accs = []
    wrong = torch.zeros(len(test_y), dtype=torch.int64)
    for seed in SEEDS:
        start = time.perf_counter()
        head = train_head(train_x, train_y, len(classes), seed)
        acc = accuracy(head, test_x, test_y)
        val_accs.append(accuracy(head, val_x, val_y))
        with torch.no_grad():
            wrong += (head(test_x).argmax(1) != test_y).long()
        accs.append(acc)
        errors = round((1 - acc) * len(test_y))
        print(
            f"  seed {seed}  test acc {acc * 100:5.1f}%  "
            f"({errors} errors)  {time.perf_counter() - start:4.1f}s"
        )
    results[name] = torch.tensor(accs)
    val_results[name] = sum(val_accs) / len(val_accs)
    wrong_counts[name] = wrong

print("\n" + "=" * 60)
print(f"SUMMARY over {len(SEEDS)} seeds, {len(test_y)} test images")
print("=" * 60)
print(f"{'Backbone':<20} {'mean':>7} {'min':>7} {'max':>7} {'std':>6} {'val':>7}")
for name, accs in results.items():
    print(
        f"{name:<20} {accs.mean() * 100:6.1f}% {accs.min() * 100:6.1f}% "
        f"{accs.max() * 100:6.1f}% {accs.std() * 100:5.1f} {val_results[name] * 100:6.1f}%"
    )
per_image = 100 / len(test_y)
print(f"\nOne test image = {per_image:.2f} percentage points")

print(f"\nPER-IMAGE ERRORS (wrong in N of {len(SEEDS)} seeds; images wrong for at least one)")
print(f"{'File':<18} {'Actual':<8}" + "".join(f"{n[:11]:>12}" for n in wrong_counts))
any_wrong = torch.stack(list(wrong_counts.values())).sum(0) > 0
for i in torch.nonzero(any_wrong).flatten().tolist():
    cells = "".join(
        f"{(str(w[i].item()) + '/' + str(len(SEEDS))) if w[i] else '-':>12}"
        for w in wrong_counts.values()
    )
    print(f"{TEST_FILES[i]:<18} {classes[test_y[i]]:<8}{cells}")
