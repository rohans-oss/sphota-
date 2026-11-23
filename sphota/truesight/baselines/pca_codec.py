"""PCA transform codec (Karhunen-Loeve transform on 8x8 blocks) with ideal entropy-coded rate.

Y (luma) and chroma are coded as in JPEG: YCbCr, chroma downsampled by 2. One PCA basis is
learned offline from TRAINING blocks only (it is shared, so it costs no bits); coefficients are
uniformly quantised and the rate is the empirical entropy of each coefficient stream, i.e. the
categorical MLE: bits = -sum_i count_i * log2(p_i).  This is an *ideal* (oracle) rate with no
header or table cost, so it is optimistic for the PCA/VQ baselines.
"""
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

PAD = 16   # pad images to a multiple of 16 so 8x8 luma blocks and 8x8 blocks of 2x-downsampled chroma tile exactly


def to_blocks(ch, b=8):
    """(H, W) -> (n_blocks, b*b), plus the (H2, W2) shape actually covered (cropped to a multiple of b)."""
    H, W = ch.shape
    H2, W2 = H // b * b, W // b * b
    x = ch[:H2, :W2]
    return x.reshape(H2 // b, b, W2 // b, b).transpose(0, 2, 1, 3).reshape(-1, b * b), (H2, W2)


def from_blocks(B, shape, b=8):
    H2, W2 = shape
    return B.reshape(H2 // b, W2 // b, b, b).transpose(0, 2, 1, 3).reshape(H2, W2)


def entropy_bits(symbols):
    """Ideal entropy-coded size in bits = -sum count * log2(p_MLE)."""
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(counts * np.log2(p)).sum())


def pad_to_multiple(a, m=PAD):
    """Edge-replicate pad the first two axes up to a multiple of m."""
    ph, pw = (-a.shape[0]) % m, (-a.shape[1]) % m
    return np.pad(a, [(0, ph), (0, pw)] + [(0, 0)] * (a.ndim - 2), mode="edge")


class PCACodec:
    """Single-channel PCA block codec."""

    def __init__(self, k=12, step=0.03, b=8):
        self.k, self.step, self.b = k, step, b

    def fit(self, train_blocks):  # blocks from TRAINING images only
        self.pca = PCA(n_components=self.k).fit(train_blocks)
        return self

    def code(self, ch, step=None):
        step = self.step if step is None else step
        B, shape = to_blocks(ch, self.b)
        c = self.pca.transform(B)
        q = np.round(c / step).astype(np.int32)                    # uniform quantiser
        bits = sum(entropy_bits(q[:, j]) for j in range(self.k))   # per-coefficient entropy model
        rec = from_blocks(self.pca.inverse_transform(q * step), shape, self.b)
        return rec, bits


class PCAImageCodec:
    """Colour wrapper: luma PCA at full resolution, one shared chroma PCA on 2x-downsampled Cb/Cr."""

    def __init__(self, k=16, b=8, chroma_step_mult=2.0):
        self.k, self.b, self.cm = k, b, chroma_step_mult
        self.luma, self.chroma = PCACodec(k, b=b), PCACodec(k, b=b)

    def fit(self, luma_blocks, chroma_blocks):
        self.luma.fit(luma_blocks)
        self.chroma.fit(chroma_blocks)
        return self

    @staticmethod
    def planes(img: Image.Image):
        """PIL RGB -> (Y, Cb_half, Cr_half) float arrays in [0, 1] (chroma 2x2-averaged), padded to PAD."""
        ycc = pad_to_multiple(np.asarray(img.convert("YCbCr"), dtype=np.float32) / 255.0)
        Y, Cb, Cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
        half = lambda c: c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).mean((1, 3))
        return Y, half(Cb), half(Cr)

    def code_image(self, img: Image.Image, step: float):
        """-> (reconstruction PIL RGB, total_bits). Rate counts all three planes."""
        W, H = img.size
        Y, Cb, Cr = self.planes(img)
        yr, yb = self.luma.code(Y, step)
        cbr, bb = self.chroma.code(Cb, step * self.cm)
        crr, br = self.chroma.code(Cr, step * self.cm)
        up = lambda c: np.asarray(Image.fromarray(c.astype(np.float32), mode="F").resize(
            (c.shape[1] * 2, c.shape[0] * 2), Image.BILINEAR))
        ycc = np.stack([yr, up(cbr), up(crr)], -1)[:H, :W]
        rec = Image.fromarray((np.clip(ycc, 0, 1) * 255 + 0.5).astype(np.uint8), "YCbCr").convert("RGB")
        return rec, yb + bb + br


def sample_blocks(files, n_images=200, per_image=2500, scale=0.5, b=8, seed=0):
    """Random luma / chroma blocks from TRAINING files (images shrunk by ``scale`` to match test-set pixel scale)."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), min(n_images, len(files)), replace=False)
    L, C = [], []
    for i in picks:
        im = Image.open(files[i]).convert("RGB")
        if scale != 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        Y, Cb, Cr = PCAImageCodec.planes(im)
        for ch, out in ((Y, L), (Cb, C), (Cr, C)):
            B, _ = to_blocks(ch, b)
            out.append(B[rng.choice(len(B), min(per_image, len(B)), replace=False)])
    return np.concatenate(L).astype(np.float32), np.concatenate(C).astype(np.float32)
