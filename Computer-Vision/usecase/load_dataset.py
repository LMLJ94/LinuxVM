import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = '/home/louise/Code/LinuxVM/Computer-Vision/images/original'
IMAGE_SHAPE = (128, 128)  # (width, height)

SPLITS = {
    'train': os.path.join(DATASET_DIR, 'Train', 'Train'),
    'test': os.path.join(DATASET_DIR, 'Test', 'Test'),
    'validation': os.path.join(DATASET_DIR, 'Validation', 'Validation'),
}


def load_split(split_path, image_shape=IMAGE_SHAPE):
    images = []
    labels = []
    classes = sorted(os.listdir(split_path))

    for label in classes:
        class_dir = os.path.join(split_path, label)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            image = cv.imread(image_path)
            if image is None:
                continue
            image = cv.resize(image, image_shape)
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


def show_images(X, y, n=9):
    n = min(n, len(X))
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv.cvtColor(X[i], cv.COLOR_BGR2RGB))
        plt.title(y[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    for split_name, split_path in SPLITS.items():
        X, y = load_split(split_path)
        print(f'{split_name}: X={X.shape} y={y.shape}')

    X, y = load_split(SPLITS['train'])
    show_images(X, y)
