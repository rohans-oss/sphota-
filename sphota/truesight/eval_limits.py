"""Keep the cost of the evaluation scripts bounded: look at a window of a large image instead of the whole of it.

The conditional U-Net has attention at 1/4 resolution, so decoding a whole image costs roughly the square of its pixel count,
and the guidance of Step 12 needs the whole image at once (it differentiates g_a over it; tiled decoding is not possible with a
guide hook). A 2K-pixel CLIC photo or a 1024 x 768 TextOCR photo is therefore many times more expensive than a Kodak image
(768 x 512). The evaluation scripts call ``limit_and_detect``: an image above ``max_pixels`` is replaced by the window of at
most ``max_pixels`` pixels (same aspect ratio, corners on a multiple of 16) that covers the most detected text / faces, so the
metrics that depend on text and faces still have something to measure. The ORIGINAL used for every metric is that same window,
so original and decode always have identical content; nothing is resized.

    max_pixels = 0 (or None)   no limit, decode whole images (what the guide describes; only for strong GPUs)
"""
import math

import numpy as np

from .protect import roi

MAX_PIXELS = 768 * 512          # the size of a Kodak image: the cost the guide's own time budget was written for
MULT = 16
MIN_BOX = 8                     # detections thinner than this after clipping to the window are dropped


def window_size(W, H, max_pixels=MAX_PIXELS, mult=MULT):
    """(w, h) of the largest window with the image's aspect ratio, sides multiples of ``mult``, w * h <= max_pixels.
    The image itself when it is already small enough."""
    if not max_pixels or W * H <= max_pixels:
        return W, H
    s = math.sqrt(max_pixels / float(W * H))
    cw, ch = max(4 * mult, int(W * s) // mult * mult), max(4 * mult, int(H * s) // mult * mult)
    cw, ch = min(cw, W), min(ch, H)
    while cw * ch > max_pixels and (cw > 4 * mult or ch > 4 * mult):
        if cw >= ch and cw > 4 * mult:
            cw -= mult
        else:
            ch -= mult
    return cw, ch


def best_window(W, H, cw, ch, boxes=(), mult=MULT):
    """Top-left corner (x0, y0), multiples of ``mult``, of the cw x ch window covering the most box area. ``boxes`` are
    (x0, y0, x1, y1) detections. Ties (and no boxes at all) go to the window closest to the image centre."""
    xs = list(range(0, W - cw + 1, mult)) or [0]
    ys = list(range(0, H - ch + 1, mult)) or [0]
    if (W - cw) % mult and xs[-1] != W - cw:
        xs.append(W - cw)
    if (H - ch) % mult and ys[-1] != H - ch:
        ys.append(H - ch)
    cx0, cy0 = (W - cw) / 2.0, (H - ch) / 2.0
    if not boxes:
        return min(xs, key=lambda v: abs(v - cx0)), min(ys, key=lambda v: abs(v - cy0))
    cover = np.zeros((H, W), dtype=np.float32)                              # area of detections per pixel (merged by max)
    for x0, y0, x1, y1 in boxes:
        cover[max(0, int(y0)):min(H, int(math.ceil(y1))), max(0, int(x0)):min(W, int(math.ceil(x1)))] = 1.0
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = cover.cumsum(0).cumsum(1)
    best, best_key = (xs[0], ys[0]), None
    for y in ys:
        for x in xs:
            s = ii[y + ch, x + cw] - ii[y, x + cw] - ii[y + ch, x] + ii[y, x]
            key = (-s, abs(x - cx0) + abs(y - cy0))
            if best_key is None or key < best_key:
                best, best_key = (x, y), key
    return best


def _shift(boxes, x0, y0, w, h, min_side=MIN_BOX):
    out = []
    for bx0, by0, bx1, by1 in boxes:
        nx0, ny0, nx1, ny1 = max(0.0, bx0 - x0), max(0.0, by0 - y0), min(float(w), bx1 - x0), min(float(h), by1 - y0)
        if nx1 - nx0 >= min_side and ny1 - ny0 >= min_side:
            out.append((nx0, ny0, nx1, ny1))
    return out


def limit_and_detect(pil, text_detector, face_detector, max_pixels=MAX_PIXELS):
    """Detect text and faces on the whole image, then (if it is above ``max_pixels``) crop to the best window.
    Returns (image, protected boxes (x, y, w, h), text boxes, face boxes, window (x0, y0, x1, y1)); every box is in the
    coordinates of the returned image. Either detector may be None."""
    W, H = pil.size
    text = [tuple(b) for b in text_detector.detect(pil)] if text_detector is not None else []
    faces = [tuple(b) for b in face_detector.detect(pil)] if face_detector is not None else []
    cw, ch = window_size(W, H, max_pixels)
    if (cw, ch) == (W, H):
        return pil, roi.plan_boxes(text + faces, H, W), text, faces, (0, 0, W, H)
    x0, y0 = best_window(W, H, cw, ch, text + faces)
    pil = pil.crop((x0, y0, x0 + cw, y0 + ch))
    text, faces = _shift(text, x0, y0, cw, ch), _shift(faces, x0, y0, cw, ch, min_side=24)
    return pil, roi.plan_boxes(text + faces, ch, cw), text, faces, (x0, y0, x0 + cw, y0 + ch)


def add_argument(ap):
    """Add the shared ``--max-pixels`` flag to an argparse parser."""
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                    help="decode at most this many pixels per image: larger images are replaced by the window that covers "
                         f"the most text / faces (default {MAX_PIXELS} = a Kodak image; 0 = whole images, very slow)")
