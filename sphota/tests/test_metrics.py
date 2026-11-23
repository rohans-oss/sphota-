import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from truesight.metrics import (bd_rate, bpp_from_bytes, cer, cosine, lpips_dist, msssim, psnr)

KODAK = Path(__file__).resolve().parents[1] / "data" / "raw" / "kodak" / "kodim01.png"


def _blur(x, k=31, sigma=8.0):
    ax = torch.arange(k) - k // 2
    g = torch.exp(-ax.float() ** 2 / (2 * sigma ** 2)); g /= g.sum()
    k2 = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k2, groups=3)


def _natural():
    """A real Kodak photo if Step 1 has downloaded it, else a smooth+edges synthetic image."""
    if KODAK.exists():
        from PIL import Image
        import torchvision.transforms as T
        return T.ToTensor()(Image.open(KODAK).convert("RGB"))[None]
    g = torch.Generator().manual_seed(0)
    x = F.interpolate(torch.rand(1, 3, 8, 12, generator=g), size=(256, 256), mode="bicubic")
    return x.clamp(0, 1)


# ---- bpp
def test_bpp():
    assert bpp_from_bytes(1000, 100, 80) == 1.0
    assert bpp_from_bytes(0, 10, 10) == 0.0
    assert math.isclose(bpp_from_bytes(12345, 512, 768), 12345 * 8 / (512 * 768))


# ---- psnr
def test_psnr_identical():
    x = torch.rand(1, 3, 64, 64)
    assert psnr(x, x) == float("inf")


def test_psnr_known():
    x = torch.zeros(1, 3, 8, 8); y = torch.full_like(x, 0.1)   # mse = 0.01
    assert abs(psnr(x, y) - 20.0) < 1e-6


def test_psnr_symmetric_and_monotonic():
    x = torch.rand(1, 3, 32, 32)
    a, b = (x + 0.01).clamp(0, 1), (x + 0.1).clamp(0, 1)
    assert psnr(x, a) == psnr(a, x) and psnr(x, a) > psnr(x, b)


# ---- ms-ssim
def test_msssim_identical_is_one():
    x = _natural()
    x = x[..., :256, :256] if x.shape[-1] >= 256 else x
    assert abs(msssim(x, x) - 1.0) < 1e-4


def test_msssim_degrades_with_noise_and_blur():
    x = _natural()[..., :256, :256]
    noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    assert msssim(x, noisy) < 0.99 and msssim(x, _blur(x)) < msssim(x, (x + 0.01 * torch.randn_like(x)).clamp(0, 1))


def test_msssim_rejects_small_and_mismatch():
    with pytest.raises(ValueError):
        msssim(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    with pytest.raises(ValueError):
        msssim(torch.rand(1, 3, 200, 200), torch.rand(1, 3, 200, 201))


# ---- lpips
def test_lpips_self_zero_and_blur_high():
    x = _natural()
    assert lpips_dist(x, x) < 1e-4
    assert lpips_dist(x, _blur(x)) > 0.2          # guide's check: heavily blurred copy is clearly different


def test_lpips_order_matches_degradation():
    x = _natural()
    mild, strong = _blur(x, 9, 1.5), _blur(x, 31, 8.0)
    assert 0 < lpips_dist(x, mild) < lpips_dist(x, strong)


# ---- cer
def test_cer():
    assert cer("5 mg", "5 mg") == 0
    assert abs(cer("5 mg", "8 mg") - 0.25) < 1e-9


def test_cer_edge_cases():
    assert cer("ABC", " abc ") == 0                # case + whitespace insensitive
    assert cer("", "") == 0 and cer("", "abc") == 3.0   # empty reference does not divide by zero
    assert cer("abc", "") == 1.0                   # everything deleted
    assert cer("4,750", "4,150") == pytest.approx(0.2)


# ---- cosine
def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 2, 3], [-1, -2, -3]) == pytest.approx(-1.0)
    assert cosine([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)       # scale invariant
    assert cosine(np.zeros(3), np.ones(3)) == 0.0                   # zero vector does not give NaN


# ---- bd-rate
R = [0.1, 0.2, 0.4, 0.8]; Q = [26, 29, 32, 35]


def test_bd_rate_same_curve():
    assert abs(bd_rate(R, Q, R, Q)) < 1e-6


def test_bd_rate_known_saving():
    # method 2 needs exactly half the bits at every quality -> -50 %
    assert bd_rate(R, Q, [r / 2 for r in R], Q) == pytest.approx(-50.0, abs=1e-6)
    # and method 1 vs 2 is the mirror: +100 %
    assert bd_rate([r / 2 for r in R], Q, R, Q) == pytest.approx(100.0, abs=1e-6)


def test_bd_rate_shifted_quality_range():
    # same log-linear curve, second method only covers part of the range -> still ~0
    assert abs(bd_rate(R, Q, R[1:] + [1.6], Q[1:] + [38])) < 1e-6


def test_bd_rate_errors():
    with pytest.raises(ValueError):
        bd_rate(R[:3], Q[:3], R, Q)
    with pytest.raises(ValueError):
        bd_rate(R, Q, R, [60, 61, 62, 63])         # no overlap
