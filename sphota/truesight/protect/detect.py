"""Text (EasyOCR / CRAFT) and face (MTCNN) detection for the protected regions (Step 11).

Both detectors return plain pixel boxes ``(x0, y0, x1, y1)`` (floats, x1 > x0, y1 > y0) in the coordinates of the input
image. The heavy models are created lazily on first use, so importing this module is cheap; either detector can be
replaced by any object with a ``detect(image) -> list[box]`` method (used by the tests and by anything faster).

    TextDetector  : EasyOCR's CRAFT *detector only* (no recognition weights needed)
    FaceDetector  : facenet_pytorch MTCNN(keep_all=True); keeps faces with probability >= 0.9 and size >= 24 px
"""
import numpy as np
from PIL import Image

class ImageMemo:
    """Remember the result of a deterministic model for images it has already seen (keyed by a hash of the pixels).

    The evaluation scripts run the same detector / OCR / face embedder on the ORIGINAL image once per method (plain, ROI, guided ...);
    the original never changes, so all but the first call are repeated work. A hit returns exactly what the model returned the first
    time, so results are identical. ``size`` bounds the memory (oldest entries are dropped)."""

    def __init__(self, size=256):
        from collections import OrderedDict
        self.size, self.d = size, OrderedDict()

    @staticmethod
    def key(*arrays):
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for a in arrays:
            a = np.ascontiguousarray(a)
            h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.digest()

    def get(self, key, compute):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = self.d[key] = compute()
        while len(self.d) > self.size:
            self.d.popitem(last=False)
        return v


MIN_FACE_PROB = 0.9
MIN_FACE_SIZE = 24          # pixels, on the shorter side of the face box


def _as_rgb_array(img):
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):          # (C, H, W) -> (H, W, C)
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8) if a.dtype.kind == "f" else a.astype(np.uint8)
    return a


def _as_pil(img):
    return img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(_as_rgb_array(img))


def text_boxes_from_easyocr(horizontal_list, free_list):
    """Convert EasyOCR's ``detect`` output for ONE image into (x0, y0, x1, y1) boxes.

    horizontal_list entries are [x_min, x_max, y_min, y_max]; free_list entries are polygons [[x, y], ...]."""
    boxes = []
    for x_min, x_max, y_min, y_max in horizontal_list:
        boxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
    for poly in free_list:
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        boxes.append((float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())))
    return [b for b in boxes if b[2] > b[0] and b[3] > b[1]]


def filter_faces(boxes, probs, min_prob=MIN_FACE_PROB, min_size=MIN_FACE_SIZE):
    """Keep faces with probability >= ``min_prob`` and both sides >= ``min_size`` px. Accepts MTCNN's raw output
    (``None`` / object arrays / NaN probabilities when nothing is found)."""
    if boxes is None or probs is None:
        return []
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    keep = []
    for b, p in zip(boxes, probs):
        if np.isfinite(p) and p >= min_prob and min(b[2] - b[0], b[3] - b[1]) >= min_size:
            keep.append(tuple(float(v) for v in b))
    return keep


class TextDetector:
    """EasyOCR text detector. ``reader`` may be passed in (to share one Reader with the OCR metric)."""

    def __init__(self, langs=("en",), gpu=False, reader=None):
        self.langs, self.gpu, self._reader = list(langs), gpu, reader
        self._memo = ImageMemo()

    @property
    def reader(self):
        if self._reader is None:
            import easyocr                                   # imported lazily: slow and optional
            self._reader = easyocr.Reader(self.langs, gpu=self.gpu, recognizer=False, verbose=False)
        return self._reader

    def detect(self, img):
        a = _as_rgb_array(img)

        def run():
            horizontal, free = self.reader.detect(a)
            return text_boxes_from_easyocr(horizontal[0], free[0])
        return self._memo.get(ImageMemo.key(a), run)


class FaceDetector:
    """MTCNN face detector (weights are bundled with the facenet-pytorch package)."""

    def __init__(self, min_prob=MIN_FACE_PROB, min_size=MIN_FACE_SIZE, device="cpu", mtcnn=None):
        self.min_prob, self.min_size, self.device, self._mtcnn = min_prob, min_size, device, mtcnn
        self._memo = ImageMemo()

    @property
    def mtcnn(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(keep_all=True, device=self.device)
        return self._mtcnn

    def detect(self, img):
        pil = _as_pil(img)
        return self._memo.get(ImageMemo.key(np.asarray(pil)), lambda: self._detect(pil))

    def _detect(self, pil):
        boxes, probs = self.mtcnn.detect(pil)
        return filter_faces(boxes, probs, self.min_prob, self.min_size)
