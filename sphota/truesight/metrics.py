"""Metrics used everywhere in TrueSight: rate, distortion, perceptual, text and face faithfulness.

Conventions: images are float tensors (B, 3, H, W) in [0, 1]. bpp = file_bits / (H * W).
"""
import math

import editdistance
import lpips
import numpy as np
import torch
from pytorch_msssim import ms_ssim

_lp = None


def _lpips():
    global _lp
    if _lp is None:
        _lp = lpips.LPIPS(net="alex", verbose=False).eval()
    return _lp


def _check_pair(x, y):
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.dim() != 4:
        raise ValueError("expected (B, C, H, W) tensors")


def bpp_from_bytes(n_bytes, h, w):
    """Bits per pixel of a file of ``n_bytes`` bytes for an ``h`` x ``w`` image."""
    return n_bytes * 8.0 / (h * w)


def psnr(x, y):
    """PSNR in dB over the whole tensor; x, y in [0, 1]. Returns inf for identical images."""
    mse = torch.mean((x.double() - y.double()) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def msssim(x, y):
    """MS-SSIM in [0, 1] (needs H, W > 160 for the default 5 scales)."""
    _check_pair(x, y)
    if min(x.shape[-2:]) <= 160:
        raise ValueError("MS-SSIM with 5 scales needs H, W > 160")
    return ms_ssim(x, y, data_range=1.0).item()


@torch.no_grad()
def lpips_dist(x, y):
    """LPIPS (AlexNet) distance, lower is better; inputs in [0, 1]."""
    _check_pair(x, y)
    m = _lpips().to(x.device)
    return m(x * 2 - 1, y * 2 - 1).mean().item()


def cer(ref: str, hyp: str):
    """Character error rate = edit distance / len(ref); case-insensitive, whitespace-trimmed."""
    ref, hyp = ref.lower().strip(), hyp.lower().strip()
    return editdistance.eval(ref, hyp) / max(1, len(ref))


def cosine(a, b):
    """Cosine similarity of two embedding vectors (face identity)."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def bd_rate(r1, q1, r2, q2):
    """Bjontegaard delta-rate (%) of method 2 vs method 1.

    r = rates (bpp), q = quality (e.g. PSNR, higher = better). Cubic fit of log-rate vs quality,
    integrated over the overlapping quality range. Negative = method 2 saves bits.
    """
    r1, q1, r2, q2 = (np.asarray(v, dtype=np.float64) for v in (r1, q1, r2, q2))
    if min(len(r1), len(r2)) < 4 or len(r1) != len(q1) or len(r2) != len(q2):
        raise ValueError("bd_rate needs at least 4 (rate, quality) points per method")
    p1, p2 = np.polyfit(q1, np.log(r1), 3), np.polyfit(q2, np.log(r2), 3)
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        raise ValueError("quality ranges do not overlap")
    P1, P2 = np.polyint(p1), np.polyint(p2)
    avg1 = (np.polyval(P1, hi) - np.polyval(P1, lo)) / (hi - lo)
    avg2 = (np.polyval(P2, hi) - np.polyval(P2, lo)) / (hi - lo)
    return (math.exp(avg2 - avg1) - 1) * 100
