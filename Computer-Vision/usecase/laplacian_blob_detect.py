import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import SPLITS
from canny_edges import sample_image_per_class
from laplacian_edges import detect_edges
from blob_detect import BLOB_SAMPLE_OVERRIDES, make_blob_detector, detect_blobs


def show_laplacian_blob_comparison(samples, detector, save_path=None):
    rows = len(samples)
    plt.figure(figsize=(12, 3 * rows))

    for row, (label, image_path) in enumerate(samples.items()):
        image = cv.imread(image_path)
        resized, mask, keypoints = detect_blobs(image, detector)
        _, laplacian_edges = detect_edges(resized)
        annotated = cv.drawKeypoints(
            resized, keypoints, np.array([]), (0, 0, 255),
            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        for col, (title, img, cmap) in enumerate([
            ('Original', cv.cvtColor(resized, cv.COLOR_BGR2RGB), None),
            ('Laplacian Edges', laplacian_edges, 'gray'),
            ('Rust Mask', mask, 'gray'),
            (f'Blobs ({len(keypoints)})', cv.cvtColor(annotated, cv.COLOR_BGR2RGB), None),
        ]):
            plt.subplot(rows, 4, row * 4 + col + 1)
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
    output_path = os.path.join(os.path.dirname(__file__), 'laplacian_blob_comparison.png')
    show_laplacian_blob_comparison(samples, detector, save_path=output_path)
