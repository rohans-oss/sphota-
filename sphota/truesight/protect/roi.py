"""Protected-region boxes (16 px aligned), ROI coding with the highest-lambda codec, feathered paste (Step 11).

Pipeline:  detected text / face boxes -> expand 10% -> align to 16 px, >= 64 x 64 -> merge overlaps -> keep the 16 largest
           -> encode every crop with the lambda = 0.0483 codec -> store (x, y, w, h, stream) in the .sphota file
Decoder:   decode the crops into an image-sized ``canvas`` and a ``mask`` (1 inside the boxes), feather the mask, paste.

The canvas and the hard mask are exactly what the ROI-lock guidance of Step 12 needs. Every box is stored as 12 bytes
(x, y, w, h, stream length) plus its stream; the ROI bytes are part of the file, so they are part of every reported bpp.
"""
import math
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..codec import bitstream as bs

MULT = 16                  # box coordinates and sizes are multiples of 16 (the file format requires it)
MIN_SIZE = 64
PAD = 0.10                 # each box is expanded by 10% so letter edges are not cut
MAX_BOXES = 16
FEATHER = 8                # pixels of soft edge in the final paste
ROI_STREAM = struct.Struct(">I")     # length of z; the ROI stream is  len(z) | z | y


# --------------------------------------------------------------------------------------------- box geometry
def _align_axis(lo, hi, length, mult=MULT, min_size=MIN_SIZE):
    """Aligned (start, size) along one axis that covers [lo, hi] when possible. start and size are multiples of
    ``mult``, the box stays inside the largest multiple of ``mult`` that fits in ``length``, and its size is at least
    ``min_size`` (or the whole axis if that is smaller). Returns None if the axis is shorter than ``mult``."""
    top = (length // mult) * mult
    if top < mult:
        return None
    lo, hi = max(0.0, lo), min(float(length), hi)
    start = math.floor(lo / mult) * mult
    end = math.ceil(hi / mult) * mult
    size = min(max(end - start, min(min_size, top)), top)
    centre = (lo + hi) / 2
    cand = round((centre - size / 2) / mult) * mult                          # centred on the content ...
    must_end = math.ceil((hi - size) / mult) * mult                          # ... but still covering it, if possible
    must_start = math.floor(lo / mult) * mult
    lo_s, hi_s = max(0, must_end), min(must_start, top - size)
    start = min(max(cand, lo_s), hi_s) if lo_s <= hi_s else min(max(cand, 0), top - size)
    return int(start), int(size)


def align_box(x0, y0, x1, y1, H, W, pad=PAD, mult=MULT, min_size=MIN_SIZE):
    """Expand a detected box by ``pad`` per side, align it to ``mult`` px, enforce ``min_size`` x ``min_size``.
    Returns (x, y, w, h) as ints inside the H x W image, or None if the image is smaller than one aligned block."""
    w, h = x1 - x0, y1 - y0
    ax = _align_axis(x0 - pad * w, x1 + pad * w, W, mult, min_size)
    ay = _align_axis(y0 - pad * h, y1 + pad * h, H, mult, min_size)
    if ax is None or ay is None:
        return None
    return ax[0], ay[0], ax[1], ay[1]


def overlaps(a, b):
    """True if the (x, y, w, h) boxes share area (touching edges do not count)."""
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def merge_boxes(boxes):
    """Merge overlapping (x, y, w, h) boxes into their bounding boxes until none overlap."""
    boxes = [tuple(int(v) for v in b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if overlaps(a, b):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = (x0, y0, x1 - x0, y1 - y0)
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def select_boxes(boxes, max_boxes=MAX_BOXES):
    """Keep the ``max_boxes`` largest boxes by area; returned in reading order (top-to-bottom, left-to-right)."""
    kept = sorted(boxes, key=lambda b: (-b[2] * b[3], b[1], b[0]))[:max_boxes]
    return sorted(kept, key=lambda b: (b[1], b[0]))


def plan_boxes(raw_boxes, H, W, pad=PAD, mult=MULT, min_size=MIN_SIZE, max_boxes=MAX_BOXES):
    """detected (x0, y0, x1, y1) boxes -> final list of aligned, merged, capped (x, y, w, h) protected boxes."""
    aligned = [a for a in (align_box(*b, H, W, pad, mult, min_size) for b in raw_boxes) if a is not None]
    return select_boxes(merge_boxes(aligned), max_boxes)


def _height_width(img):
    """(H, W) of a PIL image or an (H, W, C) array."""
    if isinstance(img, np.ndarray):
        return img.shape[0], img.shape[1]
    w, h = img.size
    return h, w


def find_protected_boxes(img, text_detector, face_detector, **kw):
    """Run both detectors on ``img`` (PIL image or (H, W, C) array) and return (final boxes, text boxes, face boxes).
    Either detector may be None. Extra keyword arguments go to plan_boxes (pad, mult, min_size, max_boxes)."""
    H, W = _height_width(img)
    text = list(text_detector.detect(img)) if text_detector is not None else []
    faces = list(face_detector.detect(img)) if face_detector is not None else []
    return plan_boxes(text + faces, H, W, **kw), text, faces


def box_area_fraction(boxes, H, W):
    """Fraction of the image covered by the boxes (they never overlap after merging)."""
    return sum(w * h for _, _, w, h in boxes) / float(H * W)


# --------------------------------------------------------------------------------------------- mask feathering
def _gauss_kernel(sigma, device):
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-(torch.arange(-r, r + 1, device=device, dtype=torch.float32) ** 2) / (2 * sigma ** 2))
    return k / k.sum(), r


def feather_mask(mask, width=FEATHER):
    """Soft version of a binary mask (B, 1, H, W): erode by ``width`` px, then Gaussian-blur with sigma = width / 2.
    The result is 1 deep inside a box, falls smoothly to ~0 over the outer ``width`` pixels *inside* the box, and is
    exactly 0 outside it - so the canvas (which only has content inside the boxes) is never blended in where it is
    empty. Image borders are not treated as edges (no seam to hide there)."""
    if width <= 0:
        return mask.clone()
    m = mask.float()
    pad = (width,) * 4
    eroded = 1 - F.max_pool2d(F.pad(1 - m, pad, mode="replicate"), 2 * width + 1, 1)
    k, r = _gauss_kernel(width / 2.0, m.device)
    s = F.pad(eroded, (r, r, 0, 0), mode="replicate")
    s = F.conv2d(s, k.view(1, 1, 1, -1))
    s = F.pad(s, (0, 0, r, r), mode="replicate")
    s = F.conv2d(s, k.view(1, 1, -1, 1))
    return s.clamp(0, 1) * m


def paste(x, canvas, soft_mask):
    """Final blend: protected pixels from the canvas, everything else from the decoded image x."""
    return soft_mask * canvas + (1 - soft_mask) * x


# --------------------------------------------------------------------------------------------- ROI streams
def pack_roi_stream(z, y):
    return ROI_STREAM.pack(len(z)) + z + y


def unpack_roi_stream(data):
    (lz,) = ROI_STREAM.unpack_from(data, 0)
    off = ROI_STREAM.size
    if off + lz > len(data):
        raise ValueError("truncated ROI stream")
    return data[off:off + lz], data[off + lz:]


@torch.no_grad()
def encode_rois(codec_hi, x, boxes):
    """Encode every box of the (1, 3, H, W) image x in [0, 1] with the high-quality codec. Returns a list of
    bitstream.Box (x, y, w, h, data). Crops are edge-padded to a multiple of 64 for the hyperprior."""
    out = []
    for X, Y, w, h in boxes:
        crop = bs.pad_to_multiple(x[..., Y:Y + h, X:X + w].to(next(codec_hi.parameters()).device), codec_hi.DOWN)
        d = codec_hi.compress(crop)
        out.append(bs.Box(X, Y, w, h, pack_roi_stream(d["z"], d["y"])))
    return out


@torch.no_grad()
def decode_rois(codec_hi, boxes, H, W, device="cpu"):
    """Decode the box streams into an image-sized canvas (B=1, 3, H, W) and a binary mask (1, 1, H, W)."""
    canvas = torch.zeros(1, 3, H, W, device=device)
    mask = torch.zeros(1, 1, H, W, device=device)
    for b in boxes:
        z, y = unpack_roi_stream(b.data)
        y_shape, z_shape = bs.latent_shapes(b.h, b.w, codec_hi.M, codec_hi.DOWN)
        crop = codec_hi.decompress({"z": z, "y": y, "z_shape": z_shape, "y_shape": y_shape})["x_hat"]
        canvas[..., b.y:b.y + b.h, b.x:b.x + b.w] = crop[..., :b.h, :b.w].to(device)
        mask[..., b.y:b.y + b.h, b.x:b.x + b.w] = 1.0
    return canvas, mask


# --------------------------------------------------------------------------------------------- whole files
@torch.no_grad()
def encode_protected(net_lo, net_hi, x, boxes, mode=0, seed=0, guidance=15):
    """x: (1, 3, H, W) in [0, 1]. Writes the .sphota bytes: the low-lambda base layer plus one high-quality stream per
    protected box. Returns (file bytes, compress() dict of the base layer)."""
    return bs.encode_image(net_lo, x, mode=mode, seed=seed, guidance=guidance, boxes=encode_rois(net_hi, x, boxes))


@dataclass
class Protected:
    """Everything the decoder needs from a protected-region file."""
    coarse: torch.Tensor                  # (1, 3, H, W) base-layer decode (no generative step)
    y_hat: torch.Tensor                   # transmitted integer latent of the base layer
    canvas: torch.Tensor                  # (1, 3, H, W) decoded protected crops (zeros elsewhere)
    mask: torch.Tensor                    # (1, 1, H, W) 1 inside the boxes (hard mask, for ROI-lock guidance)
    soft_mask: torch.Tensor               # feathered mask used for the final paste
    boxes: list = field(default_factory=list)
    sphota: object = None


@torch.no_grad()
def decode_protected(net_lo, net_hi, blob, feather=FEATHER):
    """Decode a protected-region file (bytes or path) into a Protected bundle."""
    coarse, out, s = bs.decode_image(net_lo, blob)
    canvas, mask = decode_rois(net_hi, s.boxes, s.height, s.width, device=coarse.device)
    return Protected(coarse=coarse, y_hat=out["y_hat"], canvas=canvas, mask=mask,
                     soft_mask=feather_mask(mask, feather), boxes=[(b.x, b.y, b.w, b.h) for b in s.boxes], sphota=s)


def size_breakdown(blob):
    """Where the bytes of a file go. Returns bytes and bpp for the base layer and the protected regions; the ROI
    share is the cost of faithfulness. ROI bytes = 12 per box + the box streams."""
    s = bs.unpack(blob)
    n = len(s.boxes)
    roi_streams = sum(len(b.data) for b in s.boxes)
    roi = bs.BOX.size * n + roi_streams
    total = len(blob)
    px = s.height * s.width
    return {"total_bytes": total, "base_bytes": total - roi, "roi_bytes": roi, "roi_stream_bytes": roi_streams,
            "header_bytes": bs.header_bytes(0), "n_boxes": n, "roi_share": roi / total if total else 0.0,
            "bpp_total": 8 * total / px, "bpp_base": 8 * (total - roi) / px, "bpp_roi": 8 * roi / px}
