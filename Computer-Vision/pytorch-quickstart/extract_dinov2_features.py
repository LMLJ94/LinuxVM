"""
Cache DINOv2-Small features so compare_backbones_seeds.py can test a third
backbone alongside MobileNetV3-Small and ResNet18.

Why DINOv2: MobileNet and ResNet18 tied at exactly 96.0% on every seed, which
suggests the head is not the bottleneck — the features are. Both of those were
trained with ImageNet *labels*, so they learned whatever separates "tabby cat"
from "tiger cat". DINOv2 was trained self-supervised (no labels) on 142M
curated images and is known for strong frozen features on fine texture, which
is exactly what separates a faint powdery coating from a healthy leaf.

Model: facebook/dinov2-small from Hugging Face (ViT-S/14, ~22M parameters).
Feature: the pooled output — the final-layernormed CLS token, 384 dims.

Output layout matches the other caches (features, labels, classes, backbone),
so the comparison script needs no special handling.
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel

DATA_ROOT = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/original")
CACHE_DIR = Path("feature_cache_dinov2_small")
BACKBONE_NAME = "dinov2_small"
HF_MODEL_ID = "facebook/dinov2-small"
SPLITS = {
    "train": DATA_ROOT / "Train" / "Train",
    "val": DATA_ROOT / "Validation" / "Validation",
    "test": DATA_ROOT / "Test" / "Test",
}

# Same recipe as plant_disease_transfer.py. It also happens to be exactly what
# the DINOv2 Hugging Face image processor does (shortest side 256, center crop
# 224, ImageNet mean/std), so we keep torchvision and its parallel decoding.
# 224 is divisible by the 14-pixel patch size -> a 16x16 grid of patches.
IMAGE_SIZE = 224
RESIZE_SHORT = 256
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
EXTRACT_BATCH = 32
NUM_WORKERS = 4

preprocess = transforms.Compose([
    transforms.Resize(RESIZE_SHORT),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


@torch.no_grad()
def extract(model, split_name, split_path):
    cache_file = CACHE_DIR / f"{split_name}.pt"
    if cache_file.exists():
        print(f"  {split_name:<6} already cached -> {cache_file}")
        return

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
        # pooler_output = layernorm(CLS token), shape (batch, 384)
        feats = model(pixel_values=images).pooler_output
        all_features.append(feats)
        all_labels.append(labels)
    elapsed = time.perf_counter() - start

    labels = torch.cat(all_labels)
    torch.save(
        {
            "features": torch.cat(all_features),
            "labels": labels,
            "classes": dataset.classes,
            "backbone": BACKBONE_NAME,
        },
        cache_file,
    )
    print(
        f"  {split_name:<6} extracted {len(labels):>5} images in "
        f"{elapsed:>6.1f}s ({len(labels) / elapsed:.1f} img/s) -> {cache_file}"
    )


if __name__ == "__main__":
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Loading {HF_MODEL_ID}")
    model = AutoModel.from_pretrained(HF_MODEL_ID).eval()
    print(f"Extracting features into {CACHE_DIR}/")
    for name, path in SPLITS.items():
        extract(model, name, path)
