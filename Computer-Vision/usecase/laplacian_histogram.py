import os
import cv2 as cv
import matplotlib.pyplot as plt

from laplacian_edges import sample_image_per_class, detect_edges
from load_dataset import SPLITS


def show_histogram(edges, title, save_path=None):
    hist = cv.calcHist([edges], [0], None, [256], [0, 256]).flatten()

    plt.figure(figsize=(6, 4))
    plt.bar(range(256), hist, width=1, color='black')
    plt.title(title)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Pixel count')
    plt.xlim(0, 255)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved histogram to {save_path}')
    plt.show()


if __name__ == '__main__':
    samples = sample_image_per_class(SPLITS['train'])
    out_dir = os.path.dirname(__file__)

    for label in ['Healthy', 'Powdery']:
        image = cv.imread(samples[label])
        _, edges = detect_edges(image)
        output_path = os.path.join(out_dir, f'{label.lower()}_laplacian_histogram.png')
        show_histogram(edges, f'{label} - Laplacian Histogram', save_path=output_path)
