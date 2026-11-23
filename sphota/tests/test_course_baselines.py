import numpy as np
from PIL import Image

from truesight.baselines.pca_codec import (PCAImageCodec, PCACodec, entropy_bits, from_blocks, pad_to_multiple,
                                           to_blocks)
from truesight.baselines.vq_codec import VQCodec, rgb_blocks
from truesight.metrics import bpp_from_bytes  # noqa: F401  (ensures package imports together)


def _smooth_img(w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    a = np.kron(rng.random((h // 8, w // 8, 3)), np.ones((8, 8, 1)))
    return Image.fromarray((np.clip(a + rng.normal(0, .02, a.shape), 0, 1) * 255).astype(np.uint8))


def test_block_roundtrip():
    x = np.random.rand(32, 48)
    B, shape = to_blocks(x, 8)
    assert B.shape == (24, 64) and np.allclose(from_blocks(B, shape, 8), x)
    assert np.allclose(B[0], x[:8, :8].ravel())                 # first block = top-left tile


def test_entropy_known_values():
    assert entropy_bits(np.zeros(100)) == 0.0                   # one symbol -> no information
    assert abs(entropy_bits(np.tile([0, 1, 2, 3], 25)) - 200.0) < 1e-9   # 100 symbols x 2 bits
    assert abs(entropy_bits(np.array([0, 1] * 50)) - 100.0) < 1e-9


def test_pad_to_multiple():
    assert pad_to_multiple(np.zeros((17, 33, 3))).shape == (32, 48, 3)


def test_pca_full_rank_fine_step_is_near_lossless():
    ch = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    B, _ = to_blocks(ch)
    c = PCACodec(k=64, step=1e-4).fit(B)
    rec, bits = c.code(ch)
    assert np.abs(rec - ch).max() < 1e-3 and bits > 0


def test_pca_coarser_step_fewer_bits_more_error():
    ch = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    c = PCACodec(k=16, step=0.05).fit(to_blocks(ch)[0])
    r1, b1 = c.code(ch, 0.02); r2, b2 = c.code(ch, 0.4)
    assert b2 < b1 and np.mean((r2 - ch) ** 2) > np.mean((r1 - ch) ** 2)


def test_pca_image_codec_shapes_and_monotone():
    im = _smooth_img(70, 50)                                    # not a multiple of 16: exercises padding
    L = np.concatenate([to_blocks(PCAImageCodec.planes(_smooth_img(64, 64, s))[0])[0] for s in range(4)])
    C = np.concatenate([to_blocks(PCAImageCodec.planes(_smooth_img(64, 64, s))[1])[0] for s in range(4)])
    c = PCAImageCodec(k=8).fit(L, C)
    (r1, b1), (r2, b2) = c.code_image(im, 0.05), c.code_image(im, 0.5)
    assert r1.size == im.size and r1.mode == "RGB" and b2 < b1


def test_vq_exact_when_codebook_covers_blocks():
    rng = np.random.default_rng(0)
    patterns = rng.random((4, 4, 4, 3)).astype(np.float32)                 # 4 distinct 4x4x3 blocks
    grid = rng.integers(0, 4, (8, 8))
    img = np.concatenate([np.concatenate([patterns[g] for g in row], 1) for row in grid], 0)
    B, _ = rgb_blocks(img)
    vq = VQCodec(n_codes=4).fit(B)
    rec, bits = vq.code(img)
    assert np.allclose(rec, img, atol=1e-5)
    idx = vq.km.predict(B)
    assert abs(bits - entropy_bits(idx)) < 1e-9 and bits <= 64 * 2 + 1e-9  # at most 2 bits per block


def test_vq_more_codes_lower_error():
    rng = np.random.default_rng(0)
    img = _smooth_img(64, 64)
    a = np.asarray(img, dtype=np.float32) / 255
    B, _ = rgb_blocks(a)
    errs = [np.mean((VQCodec(n).fit(B).code(a)[0] - a) ** 2) for n in (2, 16)]
    assert errs[1] < errs[0]
