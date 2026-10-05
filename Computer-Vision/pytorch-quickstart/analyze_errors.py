"""
Error analysis for the plant disease classifier.

Looks for systematic patterns in the misclassifications rather than just
reporting an accuracy number. Three questions:

  1. WHICH images were misclassified, and how confident was the model?
  2. Are errors low-confidence (borderline) or confidently wrong? Borderline
     errors suggest a decision boundary that needs sharpening; confident
     errors suggest the features themselves are misleading.
  3. Is CenterCrop discarding the diseased region? Preprocessing resizes the
     short side to 256 then crops the central 224x224, which throws away a
     substantial share of a 4000x3000 photo. If lesions sit off-center, the
     model never sees the evidence.

Outputs a contact sheet showing, for each error, the full image with the
retained crop region outlined next to the 224x224 view the model actually saw.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision import datasets

CACHE_DIR = Path("feature_cache")
TEST_DIR = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/original/Test/Test")
HEAD_PATH = Path("plant_disease_head.pth")
OUTPUT_PNG = Path("misclassified.png")

# ---------------------------------------------------------------------------
# Load cached features, the trained head, and the file paths
#
# Features were extracted with shuffle=False, so ImageFolder's sample order
# matches the cached feature order exactly.
# ---------------------------------------------------------------------------
blob = torch.load(CACHE_DIR / "test.pt", weights_only=True)
features, labels, classes = blob["features"], blob["labels"], blob["classes"]

checkpoint = torch.load(HEAD_PATH, weights_only=True)

# ---------------------------------------------------------------------------
# Guards
#
# The head is only meaningful for features from the backbone it was trained on,
# and its output positions only mean something against the class list it was
# trained with. Both files record those facts, so compare them rather than
# trusting that whoever produced them used matching settings — a label nobody
# checks protects nobody.
#
# The failures these catch are silent. ResNet18 and ResNet34 both emit 512-dim
# features, so a shape check passes while every prediction is wrong. And
# ImageFolder assigns labels alphabetically, so renaming or adding a class
# folder shifts what each output position means without changing any shape.
# ---------------------------------------------------------------------------
for key in ("backbone", "classes", "feature_dim", "preprocess"):
    if key not in checkpoint:
        raise RuntimeError(
            f"{HEAD_PATH} has no '{key}' — it predates the recipe stamp. "
            "Re-run plant_disease_transfer.py to regenerate it."
        )

if checkpoint["backbone"] != blob.get("backbone"):
    raise RuntimeError(
        f"head was trained on {checkpoint['backbone']} features, but "
        f"{CACHE_DIR}/test.pt holds {blob.get('backbone')} features."
    )

if checkpoint["classes"] != classes:
    raise RuntimeError(
        f"class order differs — head {checkpoint['classes']} vs "
        f"cache {classes}. Output position N no longer means the same disease."
    )

if checkpoint["feature_dim"] != features.shape[1]:
    raise RuntimeError(
        f"head expects {checkpoint['feature_dim']}-dim features, but the cache "
        f"holds {features.shape[1]}-dim."
    )

# Crop geometry comes from the checkpoint too, so the box drawn below is always
# the box the model actually saw.
RESIZE_SHORT = checkpoint["preprocess"]["resize_short"]
CROP_SIZE = checkpoint["preprocess"]["center_crop"]

head = nn.Linear(features.shape[1], len(classes))
head.load_state_dict(checkpoint["head"])
head.eval()

paths = [Path(p) for p, _ in datasets.ImageFolder(TEST_DIR).samples]
assert len(paths) == len(labels), "path/feature count mismatch"

with torch.no_grad():
    probabilities = head(features).softmax(dim=1)
confidence, predictions = probabilities.max(dim=1)

correct = predictions == labels
accuracy = correct.float().mean().item()

print(f"Test accuracy: {accuracy * 100:.1f}%  ({correct.sum()}/{len(labels)})")
print()

# ---------------------------------------------------------------------------
# Question 1 & 2: which errors, and how confident?
# ---------------------------------------------------------------------------
error_indices = (~correct).nonzero(as_tuple=True)[0].tolist()

print("=" * 78)
print(f"MISCLASSIFIED IMAGES ({len(error_indices)})")
print("=" * 78)
print(f"{'File':<22} {'Actual':<9} {'Predicted':<9} {'Conf':>6} {'P(actual)':>10}")
print("-" * 78)
for i in error_indices:
    true_prob = probabilities[i, labels[i]].item()
    print(
        f"{paths[i].name:<22} {classes[labels[i]]:<9} {classes[predictions[i]]:<9} "
        f"{confidence[i].item() * 100:>5.1f}% {true_prob * 100:>9.1f}%"
    )

print()
print("=" * 78)
print("CONFIDENCE ANALYSIS")
print("=" * 78)
correct_conf = confidence[correct]
error_conf = confidence[~correct]
print(f"Mean confidence, correct predictions:   {correct_conf.mean() * 100:.1f}%")
print(f"Mean confidence, incorrect predictions: {error_conf.mean() * 100:.1f}%")
print()

# How separable are errors by confidence? If most errors sit below a threshold
# that few correct predictions fall below, low-confidence flagging is viable.
for threshold in (0.60, 0.70, 0.80, 0.90):
    flagged_errors = (error_conf < threshold).sum().item()
    flagged_correct = (correct_conf < threshold).sum().item()
    print(
        f"  conf < {threshold:.0%}: catches {flagged_errors}/{len(error_conf)} errors, "
        f"false-flags {flagged_correct}/{len(correct_conf)} correct"
    )

# ---------------------------------------------------------------------------
# Question 3: how much does CenterCrop discard?
# ---------------------------------------------------------------------------
def crop_box(width, height):
    """Pixel box (in ORIGINAL image coords) that Resize+CenterCrop retains."""
    scale = RESIZE_SHORT / min(width, height)
    resized_w, resized_h = width * scale, height * scale
    left = (resized_w - CROP_SIZE) / 2 / scale
    top = (resized_h - CROP_SIZE) / 2 / scale
    side = CROP_SIZE / scale
    return left, top, side, side


print()
print("=" * 78)
print("PREPROCESSING: how much of each image survives CenterCrop?")
print("=" * 78)
sample_w, sample_h = Image.open(paths[0]).size
left, top, box_w, box_h = crop_box(sample_w, sample_h)
retained = (box_w * box_h) / (sample_w * sample_h)
print(f"Source image:  {sample_w} x {sample_h}")
print(f"Retained box:  {box_w:.0f} x {box_h:.0f} at ({left:.0f}, {top:.0f})")
print(f"Area retained: {retained * 100:.0f}%  ->  {(1 - retained) * 100:.0f}% discarded")

# ---------------------------------------------------------------------------
# Contact sheet: full image + retained region, beside the model's actual view
# ---------------------------------------------------------------------------
if error_indices:
    n = len(error_indices)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, i in enumerate(error_indices):
        image = Image.open(paths[i]).convert("RGB")
        width, height = image.size
        left, top, box_w, box_h = crop_box(width, height)

        ax = axes[row, 0]
        ax.imshow(image.resize((width // 8, height // 8)))
        ax.add_patch(
            patches.Rectangle(
                (left / 8, top / 8),
                box_w / 8,
                box_h / 8,
                linewidth=2.5,
                edgecolor="red",
                facecolor="none",
            )
        )
        ax.set_title(
            f"{paths[i].name}\nfull image — red = kept by CenterCrop",
            fontsize=9,
        )
        ax.axis("off")

        crop = image.crop((left, top, left + box_w, top + box_h)).resize(
            (CROP_SIZE, CROP_SIZE)
        )
        ax = axes[row, 1]
        ax.imshow(crop)
        ax.set_title(
            f"model's view — actual: {classes[labels[i]]}, "
            f"predicted: {classes[predictions[i]]} ({confidence[i] * 100:.0f}%)",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=85, bbox_inches="tight")
    print(f"\nContact sheet written to {OUTPUT_PNG}")
