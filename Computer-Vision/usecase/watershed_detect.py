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

FG_DIST_RATIO = 0.4  # fraction of the max distance-transform value kept as "sure foreground"
BG_DILATE_ITERATIONS = 3


def watershed_segments(image, shape=FEATURE_SHAPE):
    image = cv.resize(image, shape)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    green_mask = cv.inRange(hsv, GREEN_LOW_HSV, GREEN_HIGH_HSV)
    rust_mask = cv.inRange(hsv, RUST_LOW_HSV, RUST_HIGH_HSV)
    rust_mask = cv.morphologyEx(rust_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Same target as blob_detect.py: rust pixels, restricted to the leaf.
    leaf_mask = leaf_region_mask(green_mask, rust_mask, in_focus_mask(image))
    mask = cv.bitwise_and(rust_mask, leaf_mask)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(mask, kernel, iterations=BG_DILATE_ITERATIONS)

    dist_transform = cv.distanceTransform(mask, cv.DIST_L2, 5)
    max_dist = dist_transform.max()
    if max_dist == 0:
        # No rust pixels at all (e.g. a Healthy leaf) -> nothing to segment.
        markers = np.ones(mask.shape, dtype=np.int32)
        return image, mask, markers, 0

    _, sure_fg = cv.threshold(dist_transform, FG_DIST_RATIO * max_dist, 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1  # background becomes 1, not 0 (cv.watershed reserves 0)
    markers[unknown == 255] = 0  # contested band between fg and bg: let watershed decide

    cv.watershed(image, markers)

    lesion_count = len(set(markers.flatten()) - {-1, 1})
    return image, mask, markers, lesion_count


def colorize_markers(image, markers):
    # Cycle each label through hue so touching lesions split by watershed
    # get visibly distinct colors; boundaries (-1) are drawn in red.
    label_hue = np.uint8(179 * (markers % 180) / 180)
    saturation = np.full(label_hue.shape, 200, dtype=np.uint8)
    labeled = cv.cvtColor(cv.merge([label_hue, saturation, saturation]), cv.COLOR_HSV2BGR)

    overlay = image.copy()
    lesion_pixels = markers > 1
    blended = cv.addWeighted(image, 0.35, labeled, 0.65, 0)
    overlay[lesion_pixels] = blended[lesion_pixels]
    overlay[markers == -1] = (0, 0, 255)
    return overlay


def show_watershed_comparison(samples, save_path=None):
    rows = len(samples)
    plt.figure(figsize=(9, 3 * rows))

    for row, (label, image_path) in enumerate(samples.items()):
        image = cv.imread(image_path)
        resized, mask, markers, lesion_count = watershed_segments(image)
        overlay = colorize_markers(resized, markers)

        for col, (title, img, cmap) in enumerate([
            ('Original', cv.cvtColor(resized, cv.COLOR_BGR2RGB), None),
            ('Rust Mask', mask, 'gray'),
            (f'Watershed Segments ({lesion_count})', cv.cvtColor(overlay, cv.COLOR_BGR2RGB), None),
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
    output_path = os.path.join(os.path.dirname(__file__), 'watershed_comparison.png')
    show_watershed_comparison(samples, save_path=output_path)
