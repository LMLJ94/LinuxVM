import os
import cv2 as cv
import matplotlib.pyplot as plt

from load_dataset import SPLITS
from canny_edges import sample_image_per_class
from blob_detect import BLOB_SAMPLE_OVERRIDES
from orb_detect import make_orb_detector, detect_orb_keypoints

RATIO_THRESHOLD = 0.75  # Lowe's ratio test

# ORB descriptors are binary, so FLANN needs its LSH index rather than the
# default KD-tree index (which only handles float descriptors like SIFT/SURF).
FLANN_INDEX_LSH = 6
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
FLANN_SEARCH_PARAMS = dict(checks=50)


def other_class_samples(split_path, overrides=BLOB_SAMPLE_OVERRIDES):
    # A second image per class, distinct from the sample_image_per_class pick,
    # to match the query images against.
    samples = {}
    for label in sorted(os.listdir(split_path)):
        class_dir = os.path.join(split_path, label)
        if not os.path.isdir(class_dir):
            continue
        excluded = overrides.get(label)
        filenames = sorted(f for f in os.listdir(class_dir) if f != excluded)
        if filenames:
            samples[label] = os.path.join(class_dir, filenames[0])
    return samples


def match_descriptors(descriptors1, descriptors2, ratio=RATIO_THRESHOLD):
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) < 2:
        return []

    flann = cv.FlannBasedMatcher(FLANN_INDEX_PARAMS, FLANN_SEARCH_PARAMS)
    knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def show_match_comparison(query_samples, reference_samples, detector, save_path=None):
    classes = list(query_samples.keys())
    rows = len(classes)
    plt.figure(figsize=(14, 4 * rows))

    for row, label in enumerate(classes):
        other_label = classes[(row + 1) % len(classes)]
        query_img, _, query_kp, query_desc = detect_orb_keypoints(
            cv.imread(query_samples[label]), detector,
        )

        for col, (title, ref_label) in enumerate([
            (f'{label} vs {label} (same class)', label),
            (f'{label} vs {other_label} (different class)', other_label),
        ]):
            ref_img, _, ref_kp, ref_desc = detect_orb_keypoints(
                cv.imread(reference_samples[ref_label]), detector,
            )
            good_matches = match_descriptors(query_desc, ref_desc)
            matched = cv.drawMatches(
                query_img, query_kp, ref_img, ref_kp, good_matches, None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            plt.subplot(rows, 2, row * 2 + col + 1)
            plt.imshow(cv.cvtColor(matched, cv.COLOR_BGR2RGB))
            plt.title(f'{title} — {len(good_matches)} good matches')
            plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved comparison figure to {save_path}')
    plt.show()


if __name__ == '__main__':
    query_samples = sample_image_per_class(SPLITS['train'], overrides=BLOB_SAMPLE_OVERRIDES)
    reference_samples = other_class_samples(SPLITS['train'])
    detector = make_orb_detector()
    output_path = os.path.join(os.path.dirname(__file__), 'orb_match_comparison.png')
    show_match_comparison(query_samples, reference_samples, detector, save_path=output_path)
