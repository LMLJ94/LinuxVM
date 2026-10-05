import cv2 as cv
import numpy as np
from pathlib import Path

LOW_HSV = (0, 0, 95)
HIGH_HSV = (35, 255, 255)


def segment_by_hsv(image, low_hsv=LOW_HSV, high_hsv=HIGH_HSV):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, low_hsv, high_hsv)
    result = cv.bitwise_and(image, image, mask=mask)
    return mask, result


source_dir = Path('/home/louise/Code/LinuxVM/Computer-Vision/images')
output_root = source_dir / 'hsv_segmented'

image_paths = [
    path for path in source_dir.rglob('*.jpg')
    if output_root not in path.parents
]
print(f"Found {len(image_paths)} images.")

for i, path in enumerate(image_paths):
    image = cv.imread(str(path))
    if image is None:
        continue

    _, segmented = segment_by_hsv(image)

    relative_path = path.relative_to(source_dir)
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), segmented)

    if (i + 1) % 100 == 0:
        print(f"Segmented {i + 1}/{len(image_paths)} images.")

print("Done.")
