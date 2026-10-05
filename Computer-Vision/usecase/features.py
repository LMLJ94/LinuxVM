import cv2 as cv
import numpy as np

FEATURE_SHAPE = (256, 256)  # (width, height) — bigger than the CNN input so rust spots survive

# Rust lesions: orange/brown ring around a dark center
RUST_LOW_HSV = (5, 80, 60)
RUST_HIGH_HSV = (25, 255, 220)
RUST_MIN_SPOT_AREA = 4  # px, at FEATURE_SHAPE resolution

# Healthy leaf: saturated green
GREEN_LOW_HSV = (35, 60, 40)
GREEN_HIGH_HSV = (85, 255, 255)

FEATURE_NAMES = [
    'green_fraction',
    'rust_fraction',
    'rust_spot_count',
    'low_saturation_fraction',
    'laplacian_variance',
    'hue_hist_0', 'hue_hist_1', 'hue_hist_2', 'hue_hist_3',
    'hue_hist_4', 'hue_hist_5', 'hue_hist_6', 'hue_hist_7',
]


def extract_features(image, shape=FEATURE_SHAPE):
    image = cv.resize(image, shape)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    total_px = shape[0] * shape[1]

    green_mask = cv.inRange(hsv, GREEN_LOW_HSV, GREEN_HIGH_HSV)
    green_fraction = green_mask.mean() / 255

    rust_mask = cv.inRange(hsv, RUST_LOW_HSV, RUST_HIGH_HSV)
    rust_mask = cv.morphologyEx(rust_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    rust_fraction = rust_mask.mean() / 255
    contours, _ = cv.findContours(rust_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rust_spot_count = sum(1 for c in contours if cv.contourArea(c) >= RUST_MIN_SPOT_AREA)

    # Powdery mildew: fine whitish-gray speckle -> low saturation, not blown-out highlight
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    low_sat_mask = (saturation < 60) & (value < 240) & (value > 60)
    low_saturation_fraction = low_sat_mask.mean()

    laplacian_variance = cv.Laplacian(gray, cv.CV_64F).var()

    hue_hist = cv.calcHist([hsv], [0], None, [8], [0, 180])
    hue_hist = (hue_hist / total_px).flatten()

    return np.concatenate([
        [green_fraction, rust_fraction, rust_spot_count, low_saturation_fraction, laplacian_variance],
        hue_hist,
    ])
