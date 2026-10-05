import os
import cv2 as cv
import matplotlib.pyplot as plt

from load_dataset import SPLITS

KERNEL_SIZE = 3
CONTRAST_BOOST = 8  # Laplacian response is naturally low-contrast; brighten for display

# Hand-picked in-focus samples, chosen over the first alphabetical file per class
SAMPLE_OVERRIDES = {
    'Powdery': 'c73662f7024ef388.jpg',
}


def detect_edges(image, kernel_size=KERNEL_SIZE, contrast_boost=CONTRAST_BOOST):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv.Laplacian(blurred, cv.CV_64F, ksize=kernel_size)
    edges = cv.convertScaleAbs(laplacian, alpha=contrast_boost)
    return gray, edges


def sample_image_per_class(split_path, overrides=SAMPLE_OVERRIDES):
    samples = {}
    for label in sorted(os.listdir(split_path)):
        class_dir = os.path.join(split_path, label)
        if not os.path.isdir(class_dir):
            continue
        filename = overrides.get(label) or next(iter(os.listdir(class_dir)), None)
        if not filename:
            continue
        samples[label] = os.path.join(class_dir, filename)
    return samples


def show_laplacian_comparison(samples, save_path=None):
    rows = len(samples)
    plt.figure(figsize=(9, 3 * rows))

    for row, (label, image_path) in enumerate(samples.items()):
        image = cv.imread(image_path)
        gray, edges = detect_edges(image)

        for col, (title, img, cmap) in enumerate([
            ('Original', cv.cvtColor(image, cv.COLOR_BGR2RGB), None),
            ('Grayscale', gray, 'gray'),
            ('Laplacian', edges, 'gray'),
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
    samples = sample_image_per_class(SPLITS['train'])
    output_path = os.path.join(os.path.dirname(__file__), 'laplacian_comparison.png')
    show_laplacian_comparison(samples, save_path=output_path)
