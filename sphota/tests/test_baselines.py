import numpy as np
from PIL import Image

from truesight.baselines.jpeg import bpp_of, find_quality_for_bpp, min_file_size, pil_codec


def _img(w=256, h=192, seed=0):
    rng = np.random.default_rng(seed)
    base = np.kron(rng.integers(0, 255, (h // 8, w // 8, 3)), np.ones((8, 8, 1)))
    return Image.fromarray(np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8))


def test_bytes_grow_with_quality_and_roundtrip_size():
    im = _img()
    sizes = [pil_codec(im, q)[1] for q in (5, 30, 60, 90)]
    assert sizes == sorted(sizes) and len(set(sizes)) == 4
    rec, _ = pil_codec(im, 50)
    assert rec.size == im.size and rec.mode == "RGB"


def test_bpp_uses_real_bytes():
    im = _img()
    rec, n = pil_codec(im, 50)
    assert bpp_of(n, im) == n * 8 / (im.width * im.height)


def test_binary_search_hits_target():
    im = _img()
    for fmt in ("JPEG", "WEBP"):
        q, b, rec, ok = find_quality_for_bpp(im, 1.0, fmt)
        assert ok and abs(b - 1.0) < 0.35, (fmt, q, b)


def test_unreachable_target_flagged():
    im = _img()
    n, floor_bpp, q = min_file_size(im, "JPEG")
    got_q, b, _, ok = find_quality_for_bpp(im, floor_bpp / 10, "JPEG")
    assert not ok and abs(b - floor_bpp) < 1e-9 and got_q <= 5
