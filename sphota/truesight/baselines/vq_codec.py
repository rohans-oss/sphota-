"""k-means vector-quantisation codec on 4x4 RGB blocks (the clustering view of compression).

Each 4x4x3 block becomes the index of its nearest cluster centre (nearest neighbour, k=1). The
codebook is learned offline from TRAINING blocks and shared, so the rate is the empirical entropy
of the index stream (categorical MLE), an ideal rate without header or codebook cost.
"""
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from .pca_codec import entropy_bits, pad_to_multiple


def rgb_blocks(img, b=4):
    """img: H x W x 3 float in [0, 1] -> (n_blocks, b*b*3), plus the (H2, W2) shape covered."""
    H, W, _ = img.shape
    H2, W2 = H // b * b, W // b * b
    x = img[:H2, :W2]
    B = x.reshape(H2 // b, b, W2 // b, b, 3).transpose(0, 2, 1, 3, 4).reshape(-1, b * b * 3)
    return B, (H2, W2)


class VQCodec:
    def __init__(self, n_codes=256, b=4):
        self.n_codes, self.b = n_codes, b
        self.km = MiniBatchKMeans(n_clusters=n_codes, batch_size=8192, n_init=3, random_state=0)

    def fit(self, train_blocks):
        self.km.fit(train_blocks)
        return self

    def code(self, img):
        """img: H x W x 3 float [0,1] (multiple of b) -> (reconstruction, bits)."""
        B, (H2, W2) = rgb_blocks(img, self.b)
        idx = self.km.predict(B)                       # nearest codeword
        bits = entropy_bits(idx)
        b = self.b
        rec = self.km.cluster_centers_[idx].reshape(H2 // b, W2 // b, b, b, 3).transpose(0, 2, 1, 3, 4).reshape(H2, W2, 3)
        return rec, bits

    def code_image(self, img: Image.Image):
        W, H = img.size
        a = pad_to_multiple(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0, self.b)
        rec, bits = self.code(a)
        rec = Image.fromarray((np.clip(rec[:H, :W], 0, 1) * 255 + 0.5).astype(np.uint8))
        return rec, bits


def sample_rgb_blocks(files, n_images=200, per_image=2500, scale=0.5, b=4, seed=0):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), min(n_images, len(files)), replace=False)
    out = []
    for i in picks:
        im = Image.open(files[i]).convert("RGB")
        if scale != 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        B, _ = rgb_blocks(np.asarray(im, dtype=np.float32) / 255.0, b)
        out.append(B[rng.choice(len(B), min(per_image, len(B)), replace=False)])
    return np.concatenate(out).astype(np.float32)
