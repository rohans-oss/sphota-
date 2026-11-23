import numpy as np
import pytest
import torch
from PIL import Image

from truesight.codec.layers import PlainAE, conv, deconv, entropy_bits, round_ste
from truesight.codec.train_ae import evaluate_ae, train_ae


def _make_images(tmp_path, n=4, size=64):
    """Smooth synthetic images + list files (no dataset needed)."""
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        a = np.kron(rng.random((size // 8, size // 8, 3)), np.ones((8, 8, 1)))
        p = tmp_path / f"img{i}.png"
        Image.fromarray((a * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join(paths))
    return str(lst)


def test_conv_deconv_shapes():
    x = torch.rand(2, 3, 32, 48)
    assert conv(3, 8)(x).shape == (2, 8, 16, 24)
    assert deconv(8, 3)(conv(3, 8)(x)).shape == x.shape


@pytest.mark.parametrize("M", [8, 16, 32])
def test_plainae_shapes_and_latent_is_integer(M):
    m = PlainAE(N=16, M=M)
    x = torch.rand(2, 3, 64, 96)
    assert m.encode(x).shape == (2, M, 4, 6)
    assert m(x).shape == x.shape
    rec, q = m.reconstruct(x)
    assert q.dtype == torch.int32 and rec.min() >= 0 and rec.max() <= 1


def test_straight_through_gradient():
    y = torch.randn(10, requires_grad=True)
    out = round_ste(y)
    assert torch.equal(out.detach(), torch.round(y.detach()))      # forward: integers
    out.sum().backward()
    assert torch.equal(y.grad, torch.ones(10))                      # backward: identity


def test_gradients_reach_encoder():
    m = PlainAE(N=8, M=4)
    x = torch.rand(1, 3, 32, 32)
    ((m(x) - x) ** 2).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.enc.parameters())


def test_entropy_bits_known_values():
    assert entropy_bits(np.array([0, 1] * 8)) == pytest.approx(16.0)         # 16 fair bits
    assert entropy_bits(np.zeros(100, dtype=int)) == 0.0                     # constant -> 0 bits
    assert entropy_bits(np.array([0, 0, 0, 1])) == pytest.approx(4 * 0.8112781, rel=1e-5)
    assert entropy_bits(torch.tensor([[0, 1], [2, 3]])) == pytest.approx(8.0)  # accepts tensors


def test_training_reduces_loss_and_resume(tmp_path):
    lst = _make_images(tmp_path)
    run = tmp_path / "run"
    kw = dict(M=4, N=16, batch=2, crop=64, lr=1e-3, seed=0, log_every=5, ckpt_every=5)
    _, log1 = train_ae(lst, run, iters=40, stop_at=20, **kw)       # "killed" at 20
    assert log1[-1]["iter"] == 20
    _, log2 = train_ae(lst, run, iters=40, **kw)                   # restart -> continues from 20
    assert [r["iter"] for r in log2][:4] == [5, 10, 15, 20] and log2[-1]["iter"] == 40
    assert log2[-1]["mse"] < log2[0]["mse"]                        # loss falls
    assert all(np.isfinite(r["mse"]) for r in log2)


def test_evaluate_ae_outputs(tmp_path):
    lst = _make_images(tmp_path, n=3)
    m = PlainAE(N=16, M=4)
    res, recons = evaluate_ae(m, lst)
    assert len(recons) == 3 and recons[0].shape == (3, 64, 64)
    assert np.isfinite(res["psnr"]) and res["bpp"] >= 0 and res["latent_elems_per_pixel"] == pytest.approx(4 / 256)
