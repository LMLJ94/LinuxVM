"""
PyTorch Quickstart, CNN version.

Same FashionMNIST training/testing setup as quickstart.py, but the model
is a small convolutional network instead of a flat fully-connected one.
This is the version worth carrying over to the Plant Disease project,
since real photos have spatial structure a flat MLP throws away.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ---------------------------------------------------------------------------
# 1. Data: FashionMNIST, downloaded automatically into ./data
# ---------------------------------------------------------------------------
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
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
# 3. Model: a small CNN
#
#    Input:  1 x 28 x 28
#    Conv block 1: 1  -> 32 channels, then pool  -> 32 x 14 x 14
#    Conv block 2: 32 -> 64 channels, then pool  -> 64 x 7  x 7
#    Flatten -> fully-connected -> 10 class logits
# ---------------------------------------------------------------------------
class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits


model = ConvNeuralNetwork().to(device)
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
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# ---------------------------------------------------------------------------
# 6. Save the trained model
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "model_cnn.pth")
print("Saved PyTorch Model State to model_cnn.pth")

# ---------------------------------------------------------------------------
# 7. Reload the model and run one prediction
# ---------------------------------------------------------------------------
model = ConvNeuralNetwork().to(device)
model.load_state_dict(torch.load("model_cnn.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.unsqueeze(0).to(device)  # add batch dimension: 1x28x28 -> 1x1x28x28
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
