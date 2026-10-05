import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import SPLITS
from canny_edges import sample_image_per_class
from blob_detect import leaf_region_mask, in_focus_mask, BLOB_SAMPLE_OVERRIDES
from features import (
    FEATURE_SHAPE,
    GREEN_LOW_HSV, GREEN_HIGH_HSV,
    RUST_LOW_HSV, RUST_HIGH_HSV,
)

SIFT_N_FEATURES = 500


def make_sift_detector(n_features=SIFT_N_FEATURES):
    return cv.SIFT_create(nfeatures=n_features)


def detect_sift_keypoints(image, detector, shape=FEATURE_SHAPE):
    image = cv.resize(image, shape)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    green_mask = cv.inRange(hsv, GREEN_LOW_HSV, GREEN_HIGH_HSV)
    rust_mask = cv.inRange(hsv, RUST_LOW_HSV, RUST_HIGH_HSV)
    rust_mask = cv.morphologyEx(rust_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Restrict to the leaf (as blob_detect.py / orb_detect.py do) so SIFT
    # isn't dominated by background texture.
    leaf_mask = leaf_region_mask(green_mask, rust_mask, in_focus_mask(image))

    keypoints, descriptors = detector.detectAndCompute(gray, leaf_mask)
    return image, leaf_mask, keypoints, descriptors


def show_sift_comparison(samples, detector, save_path=None):
    rows = len(samples)
    plt.figure(figsize=(9, 3 * rows))

    for row, (label, image_path) in enumerate(samples.items()):
        image = cv.imread(image_path)
        resized, mask, keypoints, _ = detect_sift_keypoints(image, detector)
        annotated = cv.drawKeypoints(
            resized, keypoints, np.array([]), (0, 0, 255),
            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        for col, (title, img, cmap) in enumerate([
            ('Original', cv.cvtColor(resized, cv.COLOR_BGR2RGB), None),
            ('Leaf Mask', mask, 'gray'),
            (f'SIFT Keypoints ({len(keypoints)})', cv.cvtColor(annotated, cv.COLOR_BGR2RGB), None),
        ]):
            plt.subplot(rows, 3, row * 3 + col + 1)
            plt.imshow(img, cmap=cmap)
            plt.title(f'{label} - {title}' if col == 0 else title)
            plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved comparison figure to {save_path}')
    plt.show()


if __name__ == '__main__':
    samples = sample_image_per_class(SPLITS['train'], overrides=BLOB_SAMPLE_OVERRIDES)
    detector = make_sift_detector()
    output_path = os.path.join(os.path.dirname(__file__), 'sift_comparison.png')
    show_sift_comparison(samples, detector, save_path=output_path)
