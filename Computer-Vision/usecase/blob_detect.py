import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import SPLITS
from canny_edges import sample_image_per_class, SAMPLE_OVERRIDES
from laplacian_edges import detect_edges
from features import (
    FEATURE_SHAPE,
    GREEN_LOW_HSV, GREEN_HIGH_HSV,
    RUST_LOW_HSV, RUST_HIGH_HSV, RUST_MIN_SPOT_AREA,
)

# The first-alphabetical Rust file (canny_edges' default) has barely-visible
# lesions that fall outside RUST_LOW_HSV/RUST_HIGH_HSV, making the demo
# figure look like blob detection finds nothing on rust leaves. Override it
# with a sample that actually has clear pustules.
BLOB_SAMPLE_OVERRIDES = {**SAMPLE_OVERRIDES, 'Rust': '80f09587dfc7988e.jpg'}

LEAF_CLOSE_KERNEL = np.ones((15, 15), np.uint8)
EDGE_DENSITY_BOX_SIZE = 25  # local window for the in-focus vs. blurred-background split


def make_blob_detector():
    params = cv.SimpleBlobDetector_Params()

    # Rust mask is binary, so only the white (255) regions are candidate blobs
    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = RUST_MIN_SPOT_AREA
    params.maxArea = FEATURE_SHAPE[0] * FEATURE_SHAPE[1] // 4

    # Round pustules only, to tell true lesions apart from irregular mask noise
    params.filterByCircularity = True
    params.minCircularity = 0.4

    params.filterByConvexity = True
    params.minConvexity = 0.5

    params.filterByInertia = False

    return cv.SimpleBlobDetector_create(params)


def in_focus_mask(image, box_size=EDGE_DENSITY_BOX_SIZE):
    # These photos have a shallow depth of field: the leaf of interest is
    # sharp, the surrounding foliage is a blurred background of the same
    # green hue. The project's Laplacian edge map (laplacian_edges.py) is
    # dense with edges over the in-focus leaf and sparse over the blurred
    # background, so local edge density tells them apart where color can't.
    _, edges = detect_edges(image)
    local_density = cv.boxFilter(edges.astype(np.float32), -1, (box_size, box_size))
    local_density = cv.normalize(local_density, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    _, mask = cv.threshold(local_density, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return mask


def leaf_region_mask(green_mask, rust_mask, focus_mask):
    # Leaf = healthy tissue + diseased tissue, restricted to the in-focus
    # foreground and closed into one solid region so lesions near the leaf
    # edge survive while the blurred background is dropped.
    leafish = cv.bitwise_or(green_mask, rust_mask)
    combined = cv.bitwise_and(leafish, focus_mask)
    closed = cv.morphologyEx(combined, cv.MORPH_CLOSE, LEAF_CLOSE_KERNEL)

    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closed

    largest = max(contours, key=cv.contourArea)
    mask = np.zeros_like(closed)
    cv.drawContours(mask, [largest], -1, 255, cv.FILLED)
    return mask


def detect_blobs(image, detector, shape=FEATURE_SHAPE):
    image = cv.resize(image, shape)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    green_mask = cv.inRange(hsv, GREEN_LOW_HSV, GREEN_HIGH_HSV)
    rust_mask = cv.inRange(hsv, RUST_LOW_HSV, RUST_HIGH_HSV)
    rust_mask = cv.morphologyEx(rust_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    leaf_mask = leaf_region_mask(green_mask, rust_mask, in_focus_mask(image))
    mask = cv.bitwise_and(rust_mask, leaf_mask)

    keypoints = detector.detect(mask)
    return image, mask, keypoints


def show_blob_comparison(samples, detector, save_path=None):
    rows = len(samples)
    plt.figure(figsize=(9, 3 * rows))

    for row, (label, image_path) in enumerate(samples.items()):
        image = cv.imread(image_path)
        resized, mask, keypoints = detect_blobs(image, detector)
        annotated = cv.drawKeypoints(
            resized, keypoints, np.array([]), (0, 0, 255),
            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        for col, (title, img, cmap) in enumerate([
            ('Original', cv.cvtColor(resized, cv.COLOR_BGR2RGB), None),
            ('Rust Mask', mask, 'gray'),
            (f'Blobs ({len(keypoints)})', cv.cvtColor(annotated, cv.COLOR_BGR2RGB), None),
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
    detector = make_blob_detector()
    output_path = os.path.join(os.path.dirname(__file__), 'blob_comparison.png')
    show_blob_comparison(samples, detector, save_path=output_path)
