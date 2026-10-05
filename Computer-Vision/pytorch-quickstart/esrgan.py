"""
Real-ESRGAN x4 super-resolution, for the Step 8c experiment.

ESRGAN is a GAN-trained network that upscales an image 4x and invents
plausible high-frequency detail (texture, edges) that is not in the input.
Real-ESRGAN is the version trained on realistically degraded photos (blur,
noise, JPEG artefacts), which makes it the natural choice for camera images.

The network (RRDBNet: 23 "residual-in-residual dense blocks") is written out
here instead of installing the `basicsr` package, which is large and often
breaks with current torchvision. Weights are the official release:
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    sha256 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1
They are loaded with strict=True, so any mismatch between this code and the
published architecture (a wrong layer name or shape) fails loudly.
"""

from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS = Path(__file__).parent / "RealESRGAN_x4plus.pth"


class ResidualDenseBlock(nn.Module):
    """Five convs, each seeing the block input plus every earlier conv's output."""

    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.conv1 = nn.Conv2d(feat, grow, 3, 1, 1)
        self.conv2 = nn.Conv2d(feat + grow, grow, 3, 1, 1)
        self.conv3 = nn.Conv2d(feat + 2 * grow, grow, 3, 1, 1)
        self.conv4 = nn.Conv2d(feat + 3 * grow, grow, 3, 1, 1)
        self.conv5 = nn.Conv2d(feat + 4 * grow, feat, 3, 1, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(torch.cat((x, x1), 1)), 0.2)
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)), 0.2)
        x4 = F.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)), 0.2)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # residual scaling stabilises training of deep stacks


class RRDB(nn.Module):
    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(feat, grow)
        self.rdb2 = ResidualDenseBlock(feat, grow)
        self.rdb3 = ResidualDenseBlock(feat, grow)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """x4 upscaler: features at low resolution, then two 2x nearest-neighbour
    upsampling steps, each followed by a conv that fills in detail."""

    def __init__(self, feat=64, grow=32, blocks=23):
        super().__init__()
        self.conv_first = nn.Conv2d(3, feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(feat, grow) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(feat, 3, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))


_model = None


def _get_model():
    """Load once per process (DataLoader workers each get their own copy)."""
    global _model
    if _model is None:
        model = RRDBNet()
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["params_ema"]
        model.load_state_dict(state, strict=True)
        _model = model.eval()
    return _model


@torch.no_grad()
def upscale_x4(bgr):
    """Upscale a BGR uint8 image 4x with Real-ESRGAN."""
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    y = _get_model()(x).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return cv.cvtColor((y * 255.0).round().astype(np.uint8), cv.COLOR_RGB2BGR)
