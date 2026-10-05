"""
Cache Vision Transformer (ViT) features so compare_backbones_seeds.py can test
the original ViT from Google Brain alongside the CNN backbones and DINOv2.

ViT (Dosovitskiy et al., "An Image is Worth 16x16 Words", Google Brain 2020)
uses no convolutions at all. The image is cut into 16x16-pixel patches
(224x224 -> 14x14 = 196 patches), each patch is flattened and linearly
projected to a 768-dim "token", a learned [CLS] token is prepended, and 12
transformer layers let every token attend to every other token. The final
[CLS] token summarises the image; it is the feature used here.

Model: google/vit-base-patch16-224 (Hugging Face), Google's own weights:
ViT-B/16, 86.6M parameters, pretrained with labels on ImageNet-21k (14M
images), then fine-tuned on ImageNet-1k. DINOv2-Small (Step 7) is also a ViT,
but trained by Meta WITHOUT labels, so the two separate "transformer
architecture" from "how it was trained".

Two details, both handled here:

1. Which tensor is the feature. The classifier in ViTForImageClassification
   reads last_hidden_state[:, 0], the [CLS] token after the final layernorm.
   (AutoModel would add a "pooler" layer that this checkpoint has no trained
   weights for, i.e. randomly initialised: a silent trap.)
   Guard: before caching, check that the model's own classifier applied to our
   features reproduces the model's own logits.

2. ViT's input recipe differs from the ImageNet one: the whole image is
   resized to 224x224 (no centre crop, aspect ratio squashed) and pixels are
   scaled to -1..1 (mean 0.5, std 0.5). The values are read from the model's
   image processor, not retyped.
   Guard: check that the torchvision transform built from those values gives
   the same tensor as Hugging Face's own processor on a real image.

Source images: the 1024 px working copy (see extract_yolo_features.py).

Run:  python3 extract_vit_features.py
"""

import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ViTForImageClassification

WORK_ROOT = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/working_1024")
SPLITS = {
    "train": WORK_ROOT / "Train" / "Train",
    "val": WORK_ROOT / "Validation" / "Validation",
    "test": WORK_ROOT / "Test" / "Test",
}
HF_MODEL_ID = "google/vit-base-patch16-224"
BACKBONE_NAME = "vit_b16_google"
CACHE_DIR = Path("feature_cache_vit_b16_google")
EXTRACT_BATCH = 16
NUM_WORKERS = 4


def build_preprocess(processor):
    """torchvision transform built from the processor's own settings."""
    size = processor.size
    return transforms.Compose([
        transforms.Resize((size["height"], size["width"]),
                          interpolation=transforms.InterpolationMode(
                              {2: "bilinear", 3: "bicubic"}[processor.resample])),
        transforms.ToTensor(),  # rescale by 1/255, the processor's rescale_factor
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def check_preprocess(preprocess, processor, image_path):
    """Guard: our transform must match Hugging Face's processor.

    The two resize implementations round a few pixels differently: measured,
    ~0.05% of values differ, each by exactly one grey level (2/255 after the
    -1..1 scaling). So the tolerance is "at most one grey level, on at most 1%
    of values". A wrong normalisation or an added crop breaks both limits.
    """
    img = Image.open(image_path).convert("RGB")
    ours = preprocess(img)
    theirs = processor(images=img, return_tensors="pt")["pixel_values"][0]
    diff = (ours - theirs).abs()
    one_grey_level = 2 / 255 + 1e-6
    frac_differing = (diff > 1e-6).float().mean().item()
    if diff.max().item() > one_grey_level or frac_differing > 0.01:
        raise RuntimeError(
            f"preprocessing differs from the ViT processor (max diff {diff.max():.2e}, "
            f"{frac_differing:.2%} of values differ)"
        )
    return frac_differing


@torch.no_grad()
def features(model, pixels):
    # [CLS] token after the final layernorm: exactly what model.classifier reads
    return model.vit(pixel_values=pixels).last_hidden_state[:, 0]


@torch.no_grad()
def check_features(model, pixels):
    """Guard: the model's classifier on our features must give its own logits."""
    diff = (model.classifier(features(model, pixels)) - model(pixel_values=pixels).logits)
    diff = diff.abs().max().item()
    if diff > 1e-4:
        raise RuntimeError(f"features do not reproduce ViT's logits (max diff {diff:.2e})")
    return diff


@torch.no_grad()
def extract(model, preprocess, split, path):
    cache_file = CACHE_DIR / f"{split}.pt"
    if cache_file.exists():
        print(f"  {split:<6} already cached -> {cache_file}")
        return
    dataset = datasets.ImageFolder(path, transform=preprocess)
    loader = DataLoader(dataset, batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS)
    all_features, all_labels = [], []
    start = time.perf_counter()
    for i, (pixels, labels) in enumerate(loader):
        if i == 0:
            print(f"  guard passed: features reproduce ViT's logits "
                  f"(max diff {check_features(model, pixels):.1e})")
        all_features.append(features(model, pixels))
        all_labels.append(labels)
    elapsed = time.perf_counter() - start

    labels = torch.cat(all_labels)
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "features": torch.cat(all_features),
            "labels": labels,
            "classes": dataset.classes,
            "files": [Path(p).stem for p, _ in dataset.samples],
            "backbone": BACKBONE_NAME,
        },
        cache_file,
    )
    print(f"  {split:<6} extracted {len(labels):>5} images in {elapsed:>6.1f}s "
          f"({len(labels) / elapsed:.1f} img/s) -> {cache_file}")


if __name__ == "__main__":
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
    model = ViTForImageClassification.from_pretrained(HF_MODEL_ID).eval()
    for p in model.parameters():
        p.requires_grad = False
    preprocess = build_preprocess(processor)
    print(f"{HF_MODEL_ID}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print("  " + str(preprocess).replace("\n", "\n  "))
    sample = next((SPLITS["test"] / "Powdery").glob("*.jpg"))
    print(f"  guard passed: transform matches the ViT processor "
          f"({check_preprocess(preprocess, processor, sample):.2%} of values off by one grey level)")
    for split, path in SPLITS.items():
        extract(model, preprocess, split, path)
