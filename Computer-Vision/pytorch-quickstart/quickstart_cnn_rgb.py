"""
PyTorch Quickstart, RGB / variable-size version.

Same training/testing pattern as quickstart_cnn.py, but adapted for
3-channel color images of arbitrary original size — the situation you'll
actually be in with the Plant Disease dataset (real photos, not a fixed
28x28 grayscale set).

Uses CIFAR-10 (32x32 color photos, 10 classes) as a stand-in dataset,
since it's built into torchvision and downloads automatically. The two
things that changed to make this "RGB, arbitrary size, real photos"
ready are:

  1. transforms.Resize(...) added to the pipeline — real photos come in
     whatever resolution the camera/phone produced; the model needs a
     fixed input size, so every image gets resized before ToTensor().
  2. The first Conv2d takes in_channels=3 (R, G, B) instead of 1.

Swap CIFAR10 for torchvision.datasets.ImageFolder("path/to/plant_data")
to point this exact pipeline at the real plant disease photos later.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

# ---------------------------------------------------------------------------
# 1. Data: CIFAR-10, downloaded automatically into ./data
#
#    Resize to a fixed size first — this is the step that lets the same
#    pipeline handle photos of any original resolution.
# ---------------------------------------------------------------------------
IMAGE_SIZE = 64  # upscaled from CIFAR's native 32x32 to mimic resizing real photos

transform = Compose([
    Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ToTensor(),
])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 128

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# ---------------------------------------------------------------------------
# 2. Device: use GPU/MPS if available, otherwise CPU
# ---------------------------------------------------------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# ---------------------------------------------------------------------------
# 3. Model: a small CNN for 3-channel, 64x64 input
#
#    Input:  3 x 64 x 64
#    Conv block 1: 3   -> 32 channels, then pool  -> 32  x 32 x 32
#    Conv block 2: 32  -> 64 channels, then pool  -> 64  x 16 x 16
#    Conv block 3: 64  -> 128 channels, then pool -> 128 x 8  x 8
#    Flatten -> fully-connected -> 10 class logits
# ---------------------------------------------------------------------------
class ConvNeuralNetworkRGB(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits


model = ConvNeuralNetworkRGB(num_classes=10).to(device)
print(model)

# ---------------------------------------------------------------------------
# 4. Loss function and optimizer
# ---------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# ---------------------------------------------------------------------------
# 6. Save the trained model
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "model_cnn_rgb.pth")
print("Saved PyTorch Model State to model_cnn_rgb.pth")

# ---------------------------------------------------------------------------
# 7. Reload the model and run one prediction
# ---------------------------------------------------------------------------
model = ConvNeuralNetworkRGB(num_classes=10).to(device)
model.load_state_dict(torch.load("model_cnn_rgb.pth", weights_only=True))

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.unsqueeze(0).to(device)  # add batch dimension: 3x64x64 -> 1x3x64x64
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
