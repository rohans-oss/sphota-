"""Classical baselines: JPEG and WebP through Pillow, with real byte counts."""
import io

from PIL import Image

QUALITY_RANGE = {"JPEG": (1, 95), "WEBP": (0, 100)}   # q=1 is the smallest JPEG Pillow will write


def pil_codec(img: Image.Image, quality: int, fmt: str = "JPEG"):
    """Encode ``img`` with ``fmt`` at ``quality``. Returns (reconstruction RGB image, n_bytes)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=int(quality))
    n_bytes = buf.tell()
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    return rec, n_bytes


def bpp_of(n_bytes: int, img: Image.Image) -> float:
    w, h = img.size
    return n_bytes * 8.0 / (w * h)


def find_quality_for_bpp(img: Image.Image, target_bpp: float, fmt: str = "JPEG"):
    """Binary search over the quality setting to land as close as possible to ``target_bpp``.

    bpp grows monotonically with quality, so the search is O(log Q) encodes.
    Returns (quality, bpp, reconstruction, reachable). ``reachable`` is False when every quality overshoots the
    target, i.e. it is below the codec's size floor (ties prefer the lower quality).
    """
    W, H = img.size
    lo, hi = QUALITY_RANGE[fmt.upper()]
    best, ever_below = None, False
    while lo <= hi:
        q = (lo + hi) // 2
        rec, n = pil_codec(img, q, fmt)
        b = n * 8 / (W * H)
        ever_below |= b <= target_bpp
        if best is None or abs(b - target_bpp) <= abs(best[1] - target_bpp):
            best = (q, b, rec)
        if b > target_bpp:
            hi = q - 1
        else:
            lo = q + 1
    # every probed quality overshot the target -> the target is below the codec's size floor
    return (*best, ever_below)


def min_file_size(img: Image.Image, fmt: str = "JPEG"):
    """Smallest file the codec can write for ``img`` -> (bytes, bpp, quality)."""
    q = QUALITY_RANGE[fmt.upper()][0]
    _, n = pil_codec(img, q, fmt)
    return n, bpp_of(n, img), q
