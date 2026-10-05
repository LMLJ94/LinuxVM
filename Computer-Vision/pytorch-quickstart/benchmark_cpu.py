"""
CPU throughput benchmark for transfer-learning backbones.

Answers a concrete question: on THIS machine (no dedicated GPU), how long
would a pass over the Plant Disease dataset actually take, and how much
does freezing the backbone buy us?

Two scenarios are measured for each candidate backbone:

  1. Forward-only (torch.no_grad)  -> the "frozen backbone / cache the
     features once" scenario. Cheapest, and only paid once.
  2. Forward + backward + step     -> the "full fine-tuning" scenario,
     paid on every single epoch.

Models are built with weights=None on purpose. Random weights cost exactly
the same to compute as pretrained ones, so the timings are accurate and we
skip a slow download. Real training would need the pretrained weights
(downloaded once).
"""

import time

import torch
from torch import nn
from torchvision import models

torch.manual_seed(0)

print(f"PyTorch {torch.__version__}")
print(f"Threads available to torch: {torch.get_num_threads()}")
print()


def throughput(model, input_size, batch_size, train, n_warmup=1, n_timed=3):
    """Return images/second for one scenario."""
    x = torch.randn(batch_size, 3, input_size, input_size)
    y = torch.randint(0, 10, (batch_size,))
    loss_fn = nn.CrossEntropyLoss()

    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        model.eval()

    def one_batch():
        if train:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                model(x)

    for _ in range(n_warmup):  # first batch includes lazy-init overhead
        one_batch()

    start = time.perf_counter()
    for _ in range(n_timed):
        one_batch()
    elapsed = time.perf_counter() - start

    return (batch_size * n_timed) / elapsed


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Part A: backbone throughput, frozen vs fine-tuned
# ---------------------------------------------------------------------------
BATCH_SIZE = 16

candidates = [
    ("MobileNetV3-Small", lambda: models.mobilenet_v3_small(weights=None), 224),
    ("MobileNetV3-Large", lambda: models.mobilenet_v3_large(weights=None), 224),
    ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=None), 224),
    ("ResNet18", lambda: models.resnet18(weights=None), 224),
]

print("=" * 78)
print("PART A: backbone throughput (images/second)")
print("=" * 78)
print(f"{'Model':<20} {'Input':<10} {'Params':>12} {'Frozen':>12} {'Fine-tune':>12}")
print("-" * 78)

results = {}
for name, build, size in candidates:
    model = build()
    params = count_params(model)
    frozen = throughput(model, size, BATCH_SIZE, train=False)
    finetune = throughput(build(), size, BATCH_SIZE, train=True)
    results[name] = {"frozen": frozen, "finetune": finetune, "params": params}
    print(
        f"{name:<20} {f'{size}x{size}':<10} {params:>12,} "
        f"{frozen:>10.1f}/s {finetune:>10.1f}/s"
    )

print()

# ---------------------------------------------------------------------------
# Part B: training a classifier head on cached features
#
# Once the backbone is frozen and its output cached, "training" is just a
# small matrix multiply. This measures that directly.
# ---------------------------------------------------------------------------
print("=" * 78)
print("PART B: training a classifier head on cached features")
print("=" * 78)

N_SAMPLES = 20_000
FEATURE_DIM = 576  # MobileNetV3-Small's pooled feature width
NUM_CLASSES = 15
HEAD_EPOCHS = 20
HEAD_BATCH = 256

cached_features = torch.randn(N_SAMPLES, FEATURE_DIM)
cached_labels = torch.randint(0, NUM_CLASSES, (N_SAMPLES,))

head = nn.Linear(FEATURE_DIM, NUM_CLASSES)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

start = time.perf_counter()
for _ in range(HEAD_EPOCHS):
    permutation = torch.randperm(N_SAMPLES)
    for i in range(0, N_SAMPLES, HEAD_BATCH):
        idx = permutation[i : i + HEAD_BATCH]
        pred = head(cached_features[idx])
        loss = loss_fn(pred, cached_labels[idx])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
head_time = time.perf_counter() - start

print(f"{N_SAMPLES:,} cached samples, {HEAD_EPOCHS} epochs: {head_time:.1f}s total")
print(f"  -> {head_time / HEAD_EPOCHS:.2f}s per epoch")
print()

# ---------------------------------------------------------------------------
# Part C: what this means for a real dataset
# ---------------------------------------------------------------------------
print("=" * 78)
print("PART C: projected cost for a Plant Disease dataset")
print("=" * 78)

dataset_sizes = [2_000, 5_000, 20_000, 54_000]  # 54k ~= full PlantVillage
TRAINING_EPOCHS = 10


def fmt(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


for name in ("MobileNetV3-Small", "ResNet18"):
    r = results[name]
    print(f"\n{name}:")
    print(
        f"  {'Images':>8}  {'Cache once':>12}  {'+head training':>15}  "
        f"{'TOTAL':>10}  {'vs fine-tune ' + str(TRAINING_EPOCHS) + 'ep':>18}"
    )
    for n in dataset_sizes:
        cache_cost = n / r["frozen"]
        head_cost = (n / N_SAMPLES) * head_time
        total_frozen = cache_cost + head_cost
        finetune_cost = (n / r["finetune"]) * TRAINING_EPOCHS
        print(
            f"  {n:>8,}  {fmt(cache_cost):>12}  {fmt(head_cost):>15}  "
            f"{fmt(total_frozen):>10}  {fmt(finetune_cost):>18}"
        )

print()
print("Cache-once assumes no per-epoch random augmentation (features are fixed).")
print("Fine-tune column pays the full forward+backward cost on every epoch.")
