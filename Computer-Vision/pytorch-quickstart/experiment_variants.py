"""
Preprocessing experiments: does <technique> help this classifier?

The teacher's list (CLAHE, cropping, denoising, super-resolution, edges, ...)
is tested here one variant at a time. Each variant is a function
bgr_image -> bgr_image applied BEFORE the usual model preprocessing
(Resize(256) -> CenterCrop(224) -> Normalize). Everything else is held fixed:
frozen ResNet18, linear head, 40 epochs, the same 5 seeds.

Improvements over the Step-6 CenterCrop test (experiment_preprocessing.py):

  1. Five seeds instead of one, so a one-image wobble is not read as an effect.
  2. Per-image results. With 150 test images and ~6 errors, totals barely move.
     The useful question is WHICH images a variant fixes or breaks, e.g. does
     leaf cropping fix the calyx image 8ddd5ec1?
  3. One working copy. Decoding the 12-megapixel JPEGs dominates run time, so
     every image is decoded once, resized to WORK_SHORT px on the short side,
     and every variant (including the baseline) is built from that copy. The
     baseline is re-run from the copy too, so all variants share one source.

Run:  python3 experiment_variants.py            (all variants in VARIANTS)
      python3 experiment_variants.py leaf_crop  (just some)
Each variant's features are cached in feature_cache_variants/<name>/; delete a
variant's folder after changing its function.
"""

import json
import sys
import time
from pathlib import Path

import cv2 as cv
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from leaf_crop import crop_to_leaf

DATA_ROOT = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/original")
WORK_ROOT = Path("/home/louise/Code/LinuxVM/Computer-Vision/images/working_1024")
CACHE_ROOT = Path("feature_cache_variants")
RESULTS_FILE = Path("experiment_variants_results.json")
SPLITS = {
    "train": DATA_ROOT / "Train" / "Train",
    "val": DATA_ROOT / "Validation" / "Validation",
    "test": DATA_ROOT / "Test" / "Test",
}
CLASSES = ["Healthy", "Powdery", "Rust"]

WORK_SHORT = 1024  # working-copy short side; > 4x the model's 224, so no detail the model could use is lost
WORK_JPEG_QUALITY = 92
BACKBONE_NAME = "resnet18"
NUM_WORKERS = 4

# Same recipe as plant_disease_transfer.py / compare_backbones_seeds.py
HEAD_EPOCHS = 40
HEAD_BATCH = 64
HEAD_LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]

model_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Variants: bgr -> (bgr, fallback_flag)
#
# fallback_flag records "the technique could not be applied, image passed
# through unchanged". It is counted per class, because a failure rate that
# differs by class can become a shortcut the model learns.
# ---------------------------------------------------------------------------
def baseline(bgr):
    return bgr, False


def leaf_crop(bgr):
    return crop_to_leaf(bgr)


VARIANTS = {
    "baseline": baseline,
    "leaf_crop": leaf_crop,
}


# ---------------------------------------------------------------------------
# Working copy
# ---------------------------------------------------------------------------
def list_images(split):
    return [(p, CLASSES.index(p.parent.name))
            for c in CLASSES for p in sorted((SPLITS[split] / c).glob("*.jpg"))]


def _shrink_one(src):
    dst = WORK_ROOT / src.relative_to(DATA_ROOT)
    if dst.exists():
        return
    img = cv.imread(str(src))
    h, w = img.shape[:2]
    s = WORK_SHORT / min(h, w)
    img = cv.resize(img, (round(w * s), round(h * s)), interpolation=cv.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(dst), img, [cv.IMWRITE_JPEG_QUALITY, WORK_JPEG_QUALITY])


def make_working_copy():
    from multiprocessing import Pool
    sources = [p for split in SPLITS for p, _ in list_images(split)]
    todo = [p for p in sources if not (WORK_ROOT / p.relative_to(DATA_ROOT)).exists()]
    if not todo:
        return
    print(f"Building working copy: {len(todo)} images -> {WORK_ROOT}")
    start = time.perf_counter()
    with Pool(NUM_WORKERS) as pool:
        pool.map(_shrink_one, todo, chunksize=8)
    print(f"  done in {time.perf_counter() - start:.0f}s")


# ---------------------------------------------------------------------------
# Feature extraction per variant
# ---------------------------------------------------------------------------
class VariantDataset(Dataset):
    def __init__(self, split, variant_fn):
        self.items = list_images(split)
        self.variant_fn = variant_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, label = self.items[i]
        bgr = cv.imread(str(WORK_ROOT / src.relative_to(DATA_ROOT)))
        bgr, fallback = self.variant_fn(bgr)
        rgb = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        return model_preprocess(rgb), label, fallback


def _worker_init(_):
    cv.setNumThreads(1)  # 4 workers x OpenCV's own threads would oversubscribe the CPU


def build_backbone():
    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def extract(backbone, variant, split):
    cache_file = CACHE_ROOT / variant / f"{split}.pt"
    if cache_file.exists():
        blob = torch.load(cache_file, weights_only=True)
        if blob["backbone"] != BACKBONE_NAME or blob["variant"] != variant:
            raise RuntimeError(f"{cache_file} is stale; delete {cache_file.parent}/")
        return blob

    dataset = VariantDataset(split, VARIANTS[variant])
    loader = DataLoader(dataset, batch_size=32, num_workers=NUM_WORKERS,
                        worker_init_fn=_worker_init)
    feats, labels, fallbacks = [], [], []
    start = time.perf_counter()
    for images, labs, fb in loader:
        feats.append(backbone(images))
        labels.append(labs)
        fallbacks.append(fb)
    blob = {
        "features": torch.cat(feats),
        "labels": torch.cat(labels),
        "fallback": torch.cat(fallbacks),
        "files": [p.stem for p, _ in dataset.items],
        "backbone": BACKBONE_NAME,
        "variant": variant,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache_file)
    print(f"  {variant:<12} {split:<5} extracted {len(dataset):>4} images "
          f"in {time.perf_counter() - start:.0f}s")
    return blob


# ---------------------------------------------------------------------------
# Head training (identical to compare_backbones_seeds.py)
# ---------------------------------------------------------------------------
def train_head(features, labels, seed):
    torch.manual_seed(seed)
    head = nn.Linear(features.shape[1], len(CLASSES))
    optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, labels), batch_size=HEAD_BATCH, shuffle=True)
    head.train()
    for _ in range(HEAD_EPOCHS):
        for f, y in loader:
            loss = loss_fn(head(f), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return head


@torch.no_grad()
def predict(head, features):
    head.eval()
    return head(features).argmax(1)


def run_variant(backbone, variant):
    train = extract(backbone, variant, "train")
    val = extract(backbone, variant, "val")
    test = extract(backbone, variant, "test")

    test_preds = []  # [seed][image]
    val_accs = []
    for seed in SEEDS:
        head = train_head(train["features"], train["labels"], seed)
        test_preds.append(predict(head, test["features"]))
        val_accs.append((predict(head, val["features"]) == val["labels"]).float().mean().item())
    test_preds = torch.stack(test_preds)
    correct = test_preds == test["labels"]
    accs = correct.float().mean(1)

    # An image counts as an error if it is wrong in the majority of seeds
    wrong_votes = (~correct).sum(0)
    errors = {}
    for i in torch.nonzero(wrong_votes > len(SEEDS) // 2).flatten().tolist():
        predicted = test_preds[:, i].mode().values.item()
        errors[test["files"][i]] = {
            "actual": CLASSES[test["labels"][i]],
            "predicted": CLASSES[predicted],
            "wrong_in_seeds": int(wrong_votes[i]),
        }

    fallback_counts = {}
    for split, blob in [("train", train), ("val", val), ("test", test)]:
        fallback_counts[split] = {
            c: f"{int(blob['fallback'][blob['labels'] == k].sum())}/{int((blob['labels'] == k).sum())}"
            for k, c in enumerate(CLASSES)
        }

    return {
        "test_acc_per_seed": [round(a, 4) for a in accs.tolist()],
        "test_acc_mean": round(accs.mean().item(), 4),
        "val_acc_mean": round(sum(val_accs) / len(val_accs), 4),
        "errors": errors,
        "fallback": fallback_counts,
        "n_test": len(test["labels"]),
    }


def report(results):
    base = results.get("baseline")
    n = next(iter(results.values()))["n_test"]
    print("\n" + "=" * 72)
    print(f"RESULTS  ({len(SEEDS)} seeds, {n} test images, 1 image = {100 / n:.2f} pp)")
    print("=" * 72)
    print(f"{'Variant':<14} {'val':>6} {'test mean':>10} {'min':>6} {'max':>6} {'errors':>7}")
    for name, r in results.items():
        a = r["test_acc_per_seed"]
        print(f"{name:<14} {r['val_acc_mean'] * 100:5.1f}% {r['test_acc_mean'] * 100:9.1f}% "
              f"{min(a) * 100:5.1f}% {max(a) * 100:5.1f}% {len(r['errors']):>7}")

    for name, r in results.items():
        print(f"\n{name}: errors (wrong in >= {len(SEEDS) // 2 + 1}/{len(SEEDS)} seeds)")
        for f, e in r["errors"].items():
            tag = ""
            if base and name != "baseline":
                tag = "  (also baseline)" if f in base["errors"] else "  NEW"
            print(f"  {f}  {e['actual']:>7} -> {e['predicted']:<7} "
                  f"[{e['wrong_in_seeds']}/{len(SEEDS)}]{tag}")
        if base and name != "baseline":
            fixed = [f for f in base["errors"] if f not in r["errors"]]
            print(f"  fixed vs baseline: {', '.join(fixed) if fixed else 'none'}")
        if any(v.split("/")[0] != "0" for s in r["fallback"].values() for v in s.values()):
            print(f"  fallback (technique not applied), per class: {r['fallback']}")


if __name__ == "__main__":
    chosen = sys.argv[1:] or list(VARIANTS)
    make_working_copy()
    backbone = build_backbone()
    results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
    for variant in chosen:
        print(f"\nVariant: {variant}")
        results[variant] = run_variant(backbone, variant)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    report({k: results[k] for k in VARIANTS if k in results})
