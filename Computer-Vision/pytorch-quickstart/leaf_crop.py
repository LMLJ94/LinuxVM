"""
Crop a photo to its subject leaf with GrabCut.

Why GrabCut and not a colour threshold: in these photos the background is
also green foliage, so an HSV threshold selects nearly the whole frame (tested
in the step-1 preview). GrabCut starts from "the subject is roughly in the
middle" and separates foreground from background using colour models plus
edges, which isolates the leaf even against other leaves.

The mask is only used for its bounding box. Holes inside the mask (GrabCut
sometimes rejects a rust pustule or a glare patch) therefore do not matter;
what matters is that the box does not shrink and cut away leaf. Hence:

  - PAD: grow the box so edge lesions survive a slightly-too-tight mask
  - square box: the pipeline later does CenterCrop, which would otherwise cut
    the ends off a long, thin leaf
  - fallback: if the mask is implausibly small (GrabCut found nothing), return
    the full image rather than a wrong crop
"""

import cv2 as cv
import numpy as np

WORK_SIZE = 400  # GrabCut runs on a downscaled copy (~0.6s/img); box is scaled back
INIT_MARGIN = 0.10  # initial rectangle: image minus 10% on each side
GRABCUT_ITERS = 5
PAD = 0.15  # grow the box by 15% of its size on each side
MIN_AREA_FRAC = 0.05  # mask smaller than 5% of the image -> fall back to full frame
RETRY_SEEDS = range(5)  # GrabCut can collapse to an empty mask; retry before giving up


def _largest_component(mask):
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask)
    if n < 2:
        return mask
    k = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    return (labels == k).astype(np.uint8) * 255


def _grabcut_once(small_bgr, seed):
    h, w = small_bgr.shape[:2]
    mx, my = int(w * INIT_MARGIN), int(h * INIT_MARGIN)
    mask = np.zeros((h, w), np.uint8)
    bg_model, fg_model = np.zeros((1, 65)), np.zeros((1, 65))
    cv.setRNGSeed(seed)  # GrabCut's colour-model init uses k-means
    cv.grabCut(small_bgr, mask, (mx, my, w - 2 * mx, h - 2 * my),
               bg_model, fg_model, GRABCUT_ITERS, cv.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _largest_component(fg)


def leaf_mask(small_bgr):
    """GrabCut foreground mask (uint8 0/255) of the largest object.

    GrabCut's k-means initialisation sometimes lands in a solution where
    everything is background (measured: one image found the leaf on 6 of 10
    seeds, collapsed on the other 4). So retry with the next seed until the
    mask is plausibly large. Seeds are fixed, so the result is reproducible.
    """
    for seed in RETRY_SEEDS:
        mask = _grabcut_once(small_bgr, seed)
        if cv.countNonZero(mask) >= MIN_AREA_FRAC * mask.size:
            return mask
    return mask


def square_box(x, y, w, h, img_w, img_h):
    """Pad a box, make it square, and shift it to stay inside the image."""
    side = int(max(w, h) * (1 + 2 * PAD))
    side = min(side, img_w, img_h)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(cx - side / 2, 0, img_w - side))
    y0 = int(np.clip(cy - side / 2, 0, img_h - side))
    return x0, y0, side


def leaf_crop_box(bgr):
    """(x0, y0, side) in full-resolution pixels, or None to keep the full frame."""
    h, w = bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    mask = leaf_mask(small)
    if cv.countNonZero(mask) < MIN_AREA_FRAC * mask.size:
        return None
    bx, by, bw, bh = cv.boundingRect(mask)
    bx, by, bw, bh = (int(v / scale) for v in (bx, by, bw, bh))
    return square_box(bx, by, bw, bh, w, h)


def crop_to_leaf(bgr):
    """Crop to the subject leaf; returns (image, used_fallback)."""
    box = leaf_crop_box(bgr)
    if box is None:
        return bgr, True
    x0, y0, side = box
    return bgr[y0:y0 + side, x0:x0 + side], False
