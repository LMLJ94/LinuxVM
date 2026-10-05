import cv2 as cv
import numpy as np
import random
from pathlib import Path

def random_rotate(image, min_angle=-45, max_angle=45):
    angle = random.uniform(min_angle, max_angle)
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated

source_dir = Path('/home/louise/Code/LinuxVM/Computer-Vision/images')
output_root = source_dir / 'rotated'

image_paths = [
    path for path in source_dir.rglob('*.jpg')
    if output_root not in path.parents
]
print(f"Found {len(image_paths)} images.")

for i, path in enumerate(image_paths):
    image = cv.imread(str(path))
    if image is None:
        continue

    rotated = random_rotate(image)

    relative_path = path.relative_to(source_dir)
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), rotated)

    if (i + 1) % 100 == 0:
        print(f"Rotated {i + 1}/{len(image_paths)} images.")

print("Done.")
