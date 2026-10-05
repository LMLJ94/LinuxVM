import os
import cv2 as cv
import matplotlib.pyplot as plt

from load_dataset import SPLITS
from canny_edges import sample_image_per_class
from blob_detect import make_blob_detector, detect_blobs, BLOB_SAMPLE_OVERRIDES
from watershed_detect import watershed_segments, colorize_markers


def show_blob_watershed_comparison(samples, blob_detector, save_path=None):
    rows = len(samples)
    plt.figure(figsize=(12, 3 * rows))

    for row, (label, image_path) in enumerate(samples.items()):
        image = cv.imread(image_path)

        resized, mask, keypoints = detect_blobs(image, blob_detector)
        blob_annotated = cv.drawKeypoints(
            resized, keypoints, None, (0, 0, 255),
            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        _, _, markers, lesion_count = watershed_segments(image)
        watershed_overlay = colorize_markers(resized, markers)

        print(f'{label}: blob detection={len(keypoints)}  watershed={lesion_count}')

        for col, (title, img, cmap) in enumerate([
            ('Original', cv.cvtColor(resized, cv.COLOR_BGR2RGB), None),
            ('Rust Mask', mask, 'gray'),
            (f'Blob Detection ({len(keypoints)})', cv.cvtColor(blob_annotated, cv.COLOR_BGR2RGB), None),
            (f'Watershed ({lesion_count})', cv.cvtColor(watershed_overlay, cv.COLOR_BGR2RGB), None),
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
    blob_detector = make_blob_detector()
    output_path = os.path.join(os.path.dirname(__file__), 'blob_watershed_comparison.png')
    show_blob_watershed_comparison(samples, blob_detector, save_path=output_path)
