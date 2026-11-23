"""Step 8 tests: the .sphota container (pure Python) and bit-exact arithmetic coding (needs torchac, which compiles a
C++ extension: on Windows run from an 'x64 Native Tools' prompt; the torchac tests are skipped when it is unavailable)."""
from pathlib import Path

import pytest
import torch

from truesight.codec import bitstream as bs
from truesight.codec.entropy_gmm import L, gmm_cdf, to_symbols, torchac_available
from truesight.codec.train_codec import _resize_buffers
from truesight.codec.vae_codec import GMMConditional, TrueSightCodec, bpp_terms

ROOT = Path(__file__).resolve().parents[1]
needs_torchac = pytest.mark.skipif(not torchac_available(), reason="torchac cannot be imported (no C++ compiler on PATH)")


def _kodak(n):
    lst = ROOT / "data" / "processed" / "test_kodak.txt"
    if not lst.exists():
        pytest.skip("Kodak split list not found (run scripts/prepare_data.py)")
    from truesight.data import FullImageDataset
    ds = FullImageDataset(str(lst), multiple=64)
    items = [ds[i] for i in range(min(n, len(ds)))]
    if len(items) < n or not Path(ds.files[0]).exists():
        pytest.skip("Kodak images not downloaded")
    return [(x[None, :, :h, :w], p) for x, (h, w), p in items]


def _net(K=3, N=16, M=24, seed=0):
    torch.manual_seed(seed)
    return TrueSightCodec(N=N, M=M, K=K).eval()


def _clone(net):
    """A second, independent codec with the same weights - what a decoder process would build from a checkpoint."""
    n2 = TrueSightCodec(N=net.N, M=net.M, K=net.K).eval()
    n2.load_state_dict(_resize_buffers(n2, net.state_dict()))
    return n2


# ------------------------------------------------------------------ container format (pure Python)
def test_pack_unpack_roundtrip_with_boxes():
    boxes = [bs.Box(16, 32, 64, 48, b"abc"), bs.Box(0, 0, 16, 16, b"")]
    blob = bs.pack(480, 640, 3, 123456, 15, b"Z" * 7, b"Y" * 11, boxes)
    s = bs.unpack(blob)
    assert (s.height, s.width, s.mode, s.seed, s.guidance, s.guidance_scale) == (480, 640, 3, 123456, 15, 1.5)
    assert s.z == b"Z" * 7 and s.y == b"Y" * 11 and s.boxes == boxes
    assert len(blob) == bs.header_bytes(2) + 7 + 11 + 3


def test_header_layout_is_big_endian_and_documented():
    blob = bs.pack(512, 768, 1, 7, 15, b"", b"")
    assert blob[:4] == b"SPH1" and blob[4] == 1 and blob[5:7] == (512).to_bytes(2, "big") and blob[7:9] == (768).to_bytes(2, "big")
    assert bs.HEADER.size == 23 and bs.header_bytes(0) == 24 and bs.header_bytes(2) == 24 + 24


def test_file_roundtrip(tmp_path):
    p = tmp_path / "a.sphota"
    n = bs.write_file(p, 64, 64, 0, 1, 10, b"zz", b"yyy")
    assert n == p.stat().st_size and bs.read_file(p).y == b"yyy"


@pytest.mark.parametrize("mutate,msg", [
    (lambda b: b"XXXX" + b[4:], "bad magic"),
    (lambda b: b[:4] + b"\x09" + b[5:], "unsupported version"),
    (lambda b: b[:-3], "truncated"),
    (lambda b: b + b"\x00", "trailing"),
    (lambda b: b[:10], "too short|truncated"),
])
def test_unpack_rejects_corrupt_files(mutate, msg):
    blob = bs.pack(64, 64, 0, 1, 10, b"zz", b"yyyy", [bs.Box(0, 0, 16, 16, b"q")])
    with pytest.raises(ValueError, match=msg):
        bs.unpack(mutate(blob))


@pytest.mark.parametrize("kw", [
    dict(height=70000), dict(width=-1), dict(mode=256), dict(seed=2 ** 32), dict(guidance=300),
    dict(boxes=[bs.Box(8, 0, 16, 16)]),                           # not aligned to 16
    dict(boxes=[bs.Box(0, 0, 128, 16)]),                          # wider than the image
    dict(boxes=[bs.Box(0, 0, 16, 16)] * 256),                     # too many boxes
])
def test_pack_rejects_invalid_fields(kw):
    args = dict(height=64, width=64, mode=0, seed=1, guidance=10, z=b"", y=b"")
    args.update(kw)
    with pytest.raises(ValueError):
        bs.pack(args["height"], args["width"], args["mode"], args["seed"], args["guidance"], args["z"], args["y"],
                args.get("boxes", ()))


def test_latent_shapes_and_padding():
    assert bs.latent_shapes(512, 768, 192) == ((1, 192, 32, 48), (8, 12))
    assert bs.latent_shapes(500, 700, 24) == ((1, 24, 32, 44), (8, 11))         # padded up to 512 x 704
    x = torch.rand(1, 3, 100, 70)
    assert bs.pad_to_multiple(x, 64).shape == (1, 3, 128, 128)


# ------------------------------------------------------------------ CDF construction (pure torch)
def test_gmm_cdf_is_a_valid_cdf_and_matches_the_likelihood():
    torch.manual_seed(0)
    w, mu, sigma = torch.randn(1, 3, 2, 2, 2), torch.randn(1, 3, 2, 2, 2) * 4, torch.rand(1, 3, 2, 2, 2) * 3 + 0.2
    cdf = gmm_cdf(w, mu, sigma)
    assert cdf.shape == (1, 2, 2, 2, 2 * L + 2) and cdf.dtype == torch.float32 and cdf.device.type == "cpu"
    assert (cdf[..., 0] == 0).all() and (cdf[..., -1] == 1).all() and (cdf.diff(dim=-1) >= 0).all()
    y = torch.randint(-5, 6, (1, 2, 2, 2)).float()
    p_cdf = cdf.gather(-1, (y + L + 1).long().unsqueeze(-1)).squeeze(-1) - cdf.gather(-1, (y + L).long().unsqueeze(-1)).squeeze(-1)
    p_ref = GMMConditional(3).likelihood(y, w, mu, sigma)
    assert torch.allclose(p_cdf, p_ref, atol=1e-5)               # the coder's probabilities equal the training likelihood


def test_gmm_cdf_survives_degenerate_parameters():
    w, mu, sigma = torch.zeros(1, 1, 1, 1, 1), torch.full((1, 1, 1, 1, 1), 1e4), torch.full((1, 1, 1, 1, 1), 1e-6)
    cdf = gmm_cdf(w, mu, sigma)                                   # all mass far outside [-L, L]
    assert torch.isfinite(cdf).all() and cdf[..., -1].item() == 1.0 and (cdf.diff(dim=-1) >= 0).all()


def test_to_symbols_clamps_to_range():
    s = to_symbols(torch.tensor([-1000.0, -60.0, 0.0, 60.0, 1000.0]))
    assert s.dtype == torch.int16 and s.tolist() == [0, 0, L, 2 * L, 2 * L]


# ------------------------------------------------------------------ arithmetic coding (torchac)
@needs_torchac
def test_encode_decode_y_roundtrip_incl_boundaries():
    from truesight.codec.entropy_gmm import decode_y, encode_y
    torch.manual_seed(0)
    w, mu, sigma = torch.randn(1, 3, 6, 5, 7), torch.randn(1, 3, 6, 5, 7) * 6, torch.rand(1, 3, 6, 5, 7) * 5 + 0.3
    y = torch.randint(-L, L + 1, (1, 6, 5, 7)).float()
    y[0, 0, 0, :2] = torch.tensor([-L, L])                        # extreme symbols, even where the model gives them ~0 mass
    b = encode_y(y, w, mu, sigma)
    assert isinstance(b, bytes) and torch.equal(decode_y(b, w, mu, sigma), y)


@needs_torchac
def test_five_kodak_images_decode_bit_exactly_through_files(tmp_path):
    net, dec = _net(), None
    dec = _clone(net)                                            # independent decoder with the same weights
    for k, (x, path) in enumerate(_kodak(5)):
        blob, d = bs.encode_image(net, x, mode=2, seed=k)
        f = tmp_path / f"k{k}.sphota"
        f.write_bytes(blob)
        x_hat, out, s = bs.decode_image(dec, f)                  # decode from the file, with the other codec instance
        h, w = x.shape[-2:]
        xp = bs.pad_to_multiple(x, 64)
        expected = torch.round(net.g_a(xp)).clamp(-L, L)          # transmitted latent = clamp(round(g_a(x)), -L, L)
        assert torch.equal(out["y_hat"], expected), path
        assert torch.equal(out["y_hat"], d["y_hat"])
        assert x_hat.shape == x.shape and (s.height, s.width, s.mode, s.seed) == (h, w, 2, k)
        assert torch.equal(x_hat, dec.g_s(out["y_hat"]).clamp(0, 1)[..., :h, :w])


@needs_torchac
def test_real_size_matches_the_estimate_within_3_percent():
    net = _net()
    for x, path in _kodak(3):
        h, w = x.shape[-2:]
        blob, d = bs.encode_image(net, x)
        payload_bits = 8 * (len(d["z"]) + len(d["y"]))
        est_bits = sum(bpp_terms(net(bs.pad_to_multiple(x, 64))["likelihoods"], 1).values()).item()   # total bits
        assert abs(payload_bits / est_bits - 1) < 0.03, (path, payload_bits, est_bits)
        assert len(blob) == payload_bits // 8 + bs.header_bytes(0)


@needs_torchac
def test_latents_beyond_the_symbol_range_are_clamped_and_reported():
    net = _net()
    with torch.no_grad():
        net.g_a[-1].weight.mul_(400)                              # blow up the encoder output well past +-60
    x = torch.rand(1, 3, 64, 64)
    blob, d = bs.encode_image(net, x)
    assert d["n_clamped"] > 0 and d["y_hat"].abs().max() <= L
    _, out, _ = bs.decode_image(_clone(net), blob)
    assert torch.equal(out["y_hat"], d["y_hat"])                  # the clamped latent is exactly what is decoded


@needs_torchac
def test_k1_codec_roundtrip_and_non_multiple_of_64_size():
    net = _net(K=1)
    x = torch.rand(1, 3, 100, 90)                                 # padded to 128 x 128 internally
    blob, d = bs.encode_image(net, x)
    x_hat, out, s = bs.decode_image(_clone(net), blob)
    assert x_hat.shape == x.shape and (s.height, s.width) == (100, 90) and torch.equal(out["y_hat"], d["y_hat"])


@needs_torchac
def test_compress_guards_and_corruption_is_detected():
    net = _net()
    with pytest.raises(ValueError):
        net.compress(torch.rand(2, 3, 64, 64))                    # batch size
    with pytest.raises(ValueError):
        net.compress(torch.rand(1, 3, 70, 64))                    # not a multiple of 64
    x = torch.rand(1, 3, 64, 64)
    d = net.compress(x)
    bad = dict(d, y=bytes([d["y"][0] ^ 0xFF]) + d["y"][1:])       # flip the first byte: the decoder must not agree
    assert not torch.equal(net.decompress(bad)["y_hat"], d["y_hat"])
