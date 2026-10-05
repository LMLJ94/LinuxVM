"""
Cache YOLOv8 classification features so compare_backbones_seeds.py can test
YOLO as a frozen backbone alongside MobileNet, ResNet18 and DINOv2.

Models: yolov8n-cls (2.7M parameters) and yolov8s-cls (6.4M), Ultralytics,
ImageNet-pretrained. Feature: the 1280-dim pooled vector just before the
classifier's final Linear layer.

Two traps, both handled here:

1. The usual trick (replace the last Linear with Identity) is WRONG for YOLO.
   In eval mode its Classify head applies softmax itself, so Identity would
   return softmaxed "probabilities" instead of features. Instead the network is
   rebuilt by hand: every layer except the head, then the head's conv and
   global pool, and stop before its Linear.
   Guard: before caching, the script checks that Linear + softmax on these
   features reproduces the full model's own output exactly. If Ultralytics ever
   changes the head, this fails loudly instead of caching wrong features.

2. YOLO was trained on 0-1 pixels with NO ImageNet mean/std normalisation, and
   resizes the short side straight to 224 (no 256 step). The recipe is read
   from the model itself (model.transforms), not retyped.

Source images: the 1024 px working copy made by experiment_variants.py. It
skips the 12-megapixel JPEG decode that dominates extraction time, and Step 8
verified it gives ResNet18 exactly the same six errors as the originals.

Run:  python3 extract_yolo_features.py               (both models)
      python3 extract_yolo_features.py yolov8n_cls   (one)
"""

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from ultralytics import YOLO

WORK_ROOT = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/working_1024")
SPLITS = {
    "train": WORK_ROOT / "Train" / "Train",
    "val": WORK_ROOT / "Validation" / "Validation",
    "test": WORK_ROOT / "Test" / "Test",
}
MODELS = {
    "yolov8n_cls": "yolov8n-cls.pt",
    "yolov8s_cls": "yolov8s-cls.pt",
}
EXTRACT_BATCH = 32
NUM_WORKERS = 4


def build_extractor(weights):
    """Frozen YOLO feature extractor, plus the full model and its preprocessing."""
    yolo_model = YOLO(weights).model.eval()
    net = yolo_model.model  # nn.Sequential of layers; the last one is Classify
    head = net[-1]
    extractor = nn.Sequential(*net[:-1], head.conv, head.pool, nn.Flatten(1)).eval()
    for p in yolo_model.parameters():
        p.requires_grad = False
    return extractor, yolo_model, head.linear, yolo_model.transforms


@torch.no_grad()
def check_extractor(extractor, full_model, linear, images):
    """Guard: Linear + softmax on our features must equal the model's own output."""
    out = full_model(images)
    full_probs = out[0] if isinstance(out, (tuple, list)) else out
    rebuilt_probs = linear(extractor(images)).softmax(1)
    max_diff = (full_probs - rebuilt_probs).abs().max().item()
    if max_diff > 1e-5:
        raise RuntimeError(
            f"extracted features do not reproduce YOLO's own output "
            f"(max difference {max_diff:.2e}); the head structure has changed"
        )
    return max_diff


@torch.no_grad()
def extract(name, extractor, full_model, linear, preprocess, split, path):
    cache_dir = Path(f"feature_cache_{name}")
    cache_file = cache_dir / f"{split}.pt"
    if cache_file.exists():
        print(f"  {split:<6} already cached -> {cache_file}")
        return

    dataset = datasets.ImageFolder(path, transform=preprocess)
    loader = DataLoader(dataset, batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS)
    all_features, all_labels = [], []
    start = time.perf_counter()
    for i, (images, labels) in enumerate(loader):
        if i == 0:
            diff = check_extractor(extractor, full_model, linear, images)
            print(f"  guard passed: features reproduce YOLO's output (max diff {diff:.1e})")
        all_features.append(extractor(images))
        all_labels.append(labels)
    elapsed = time.perf_counter() - start

    labels = torch.cat(all_labels)
    cache_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "features": torch.cat(all_features),
            "labels": labels,
            "classes": dataset.classes,
            "files": [Path(p).stem for p, _ in dataset.samples],
            "backbone": name,
        },
        cache_file,
    )
    print(f"  {split:<6} extracted {len(labels):>5} images in {elapsed:>6.1f}s "
          f"({len(labels) / elapsed:.1f} img/s) -> {cache_file}")


if __name__ == "__main__":
    if not WORK_ROOT.exists():
        sys.exit(f"{WORK_ROOT} missing: run experiment_variants.py once to build it")
    for name in sys.argv[1:] or list(MODELS):
        extractor, full_model, linear, preprocess = build_extractor(MODELS[name])
        n_params = sum(p.numel() for p in full_model.parameters()) / 1e6
        print(f"\n{name}: {n_params:.1f}M parameters, recipe from the model:")
        print("  " + str(preprocess).replace("\n", "\n  "))
        for split, path in SPLITS.items():
            extract(name, extractor, full_model, linear, preprocess, split, path)
