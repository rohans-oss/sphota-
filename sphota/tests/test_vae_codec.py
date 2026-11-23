import math

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from truesight.codec.eval_codec import ablation_k, bd_table, beats_baseline, evaluate_codec
from truesight.codec.train_codec import load_codec, lr_at, train_codec
from truesight.codec.vae_codec import (LAMBDAS, SCALE_MIN, GMMConditional, TrueSightCodec, bpp_terms, rd_loss)

N, M = 16, 24


def _net(K=3):
    torch.manual_seed(0)
    return TrueSightCodec(N=N, M=M, K=K)


def _images(tmp_path, n=4, size=128):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        a = np.kron(rng.random((size // 16, size // 16, 3)), np.ones((16, 16, 1)))
        p = tmp_path / f"i{i}.png"
        Image.fromarray((a * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join(paths))
    return str(lst)


# ---- GMM likelihood
def test_gmm_probabilities_sum_to_one_over_integers():
    g = GMMConditional(K=3)
    torch.manual_seed(1)
    w, mu, sigma = torch.randn(1, 3, 1, 1, 1), torch.randn(1, 3, 1, 1, 1) * 3, torch.rand(1, 3, 1, 1, 1) * 4 + 0.5
    ys = torch.arange(-80, 81).float().view(-1, 1, 1, 1)
    p = torch.stack([g.likelihood(y.view(1, 1, 1, 1), w, mu, sigma) for y in ys]).sum()
    assert p.item() == pytest.approx(1.0, abs=1e-4)


def test_gmm_k1_equals_discretised_gaussian():
    g = GMMConditional(K=1)
    y = torch.tensor([[[[2.0]]]])
    mu, sigma = torch.full((1, 1, 1, 1, 1), 0.7), torch.full((1, 1, 1, 1, 1), 1.3)
    n = torch.distributions.Normal(0.7, 1.3)
    ref = (n.cdf(torch.tensor(2.5)) - n.cdf(torch.tensor(1.5))).item()
    assert g.likelihood(y, torch.zeros(1, 1, 1, 1, 1), mu, sigma).item() == pytest.approx(ref, rel=1e-5)


def test_gmm_scale_floor_and_prob_floor():
    g = GMMConditional(K=1)
    y = torch.zeros(1, 1, 1, 1)
    tiny = torch.full((1, 1, 1, 1, 1), 1e-6)
    p = g.likelihood(y, torch.zeros_like(tiny), torch.zeros_like(tiny), tiny)
    ref = torch.distributions.Normal(0.0, SCALE_MIN)
    assert p.item() == pytest.approx((ref.cdf(torch.tensor(.5)) - ref.cdf(torch.tensor(-.5))).item(), rel=1e-5)
    far = g.likelihood(torch.full((1, 1, 1, 1), 1e4), torch.zeros_like(tiny), torch.zeros_like(tiny), torch.ones_like(tiny))
    assert far.item() == pytest.approx(1e-9)                          # floored, so -log never gives inf


# ---- model
def test_forward_shapes_train_and_eval():
    net, x = _net(), torch.rand(2, 3, 128, 64)
    out = net(x)
    assert out["x_hat"].shape == x.shape and out["y"].shape == (2, M, 8, 4)
    assert out["likelihoods"]["y"].shape == (2, M, 8, 4) and out["likelihoods"]["z"].shape == (2, N, 2, 1)
    net.eval()
    out = net(x)
    assert torch.equal(out["y_hat"], torch.round(out["y"]))           # eval path uses true integers


def test_rejects_non_multiple_of_64():
    with pytest.raises(ValueError):
        _net()(torch.rand(1, 3, 96, 64))


def test_gmm_param_shapes_and_positive_scale():
    net = _net()
    y = torch.randn(1, M, 4, 4)
    w, mu, sigma = net.gmm_params(torch.randn(1, N, 1, 1), y.shape)
    assert w.shape == mu.shape == sigma.shape == (1, 3, M, 4, 4) and (sigma > 0).all()


def test_rd_loss_matches_formula_and_backprops_everywhere():
    net, x = _net(), torch.rand(2, 3, 64, 64)
    out = net(x)
    loss, bpp, mse = rd_loss(out, x, 0.013)
    n_px = 2 * 64 * 64
    manual = sum(-torch.log2(l).sum() for l in out["likelihoods"].values()) / n_px
    assert bpp.item() == pytest.approx(manual.item(), rel=1e-5) and bpp.item() > 0
    assert loss.item() == pytest.approx(bpp.item() + 0.013 * 255 ** 2 * mse.item(), rel=1e-5)
    loss.backward()
    main, aux = net.split_parameters()
    missing = [n for n, p in net.named_parameters() if p.grad is None and not n.endswith(".quantiles")]
    assert not missing, missing                                       # encoder, decoder, hyper nets, GDN all get gradient


def test_split_parameters_partition():
    net = _net()
    main, aux = net.split_parameters()
    assert len(aux) == 1 and len(main) + len(aux) == len(list(net.parameters()))
    assert net.aux_loss().ndim == 0 and torch.isfinite(net.aux_loss())


def test_bpp_terms_split():
    net, x = _net(), torch.rand(1, 3, 64, 64)
    out = net(x)
    t = bpp_terms(out["likelihoods"], 64 * 64)
    assert set(t) == {"y", "z"} and all(v > 0 for v in t.values())


def test_lambdas_are_the_guide_values():
    assert LAMBDAS == (0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483)


def test_lr_schedule_drops_at_90_percent():
    assert lr_at(0, 1000, 1e-4) == 1e-4 and lr_at(899, 1000, 1e-4) == 1e-4
    assert lr_at(900, 1000, 1e-4) == pytest.approx(1e-5)


# ---- training / resume / load / eval
def test_training_resume_final_and_load(tmp_path):
    lst = _images(tmp_path)
    run = tmp_path / "run"
    kw = dict(K=3, N=N, M=M, batch=2, crop=64, workers=0, train_list=lst, log_every=2, ckpt_every=4, device="cpu")
    train_codec(0.013, run, iters=12, stop_at=6, **kw)                  # "killed" at 6
    assert (run / "last.pt").exists() and not (run / "final.pt").exists()
    train_codec(0.013, run, iters=12, **kw)                              # restart -> finishes
    assert (run / "final.pt").exists()
    log = pd.read_csv(run / "log.csv")
    assert log["it"].tolist() == [2, 4, 6, 8, 10, 12]                    # continued from 6, no restart from 0
    assert np.isfinite(log[["loss", "bpp", "mse", "aux"]].to_numpy()).all()
    net, cfg = load_codec(run / "final.pt")
    assert cfg == dict(N=N, M=M, K=3, lmbda=0.013, act="gdn") and not net.training
    res = evaluate_codec(net, lst, max_images=2, with_perceptual=False)
    assert len(res) == 2 and (res.bpp > 0).all() and np.isfinite(res.psnr).all()
    assert (res.bpp - (res.bpp_y + res.bpp_z)).abs().max() < 1e-6


def test_finetune_from_checkpoint(tmp_path):
    lst = _images(tmp_path)
    kw = dict(N=N, M=M, batch=2, crop=64, workers=0, train_list=lst, log_every=2, ckpt_every=2, device="cpu")
    train_codec(0.013, tmp_path / "base", K=3, iters=2, **kw)
    train_codec(0.0035, tmp_path / "ft", K=3, iters=2, init_from=str(tmp_path / "base" / "final.pt"), **kw)
    assert (tmp_path / "ft" / "final.pt").exists()
    with pytest.raises(ValueError):                                      # K mismatch must not silently load
        train_codec(0.0035, tmp_path / "bad", K=1, iters=2, init_from=str(tmp_path / "base" / "final.pt"), **kw)


def test_training_loss_goes_down(tmp_path):
    lst = _images(tmp_path, n=4)
    run = tmp_path / "r"
    train_codec(0.0483, run, K=3, N=N, M=M, iters=60, batch=4, crop=64, workers=0, train_list=lst, log_every=10,
                ckpt_every=60, lr=1e-3, device="cpu")
    log = pd.read_csv(run / "log.csv")
    assert log.loss.iloc[-1] < log.loss.iloc[0] and np.isfinite(log.to_numpy()).all()


# ---- evaluation helpers
def test_bd_table_and_beats_baseline():
    base = pd.DataFrame(dict(bpp=[0.1, 0.2, 0.4, 0.8], psnr=[24., 27., 30., 33.]))
    better = pd.DataFrame(dict(bpp=[0.05, 0.1, 0.2, 0.4], psnr=[24., 27., 30., 33.]))   # half the bits
    assert bd_table(better, {"B": base})["B"] == pytest.approx(-50.0, abs=1.0)
    ok, det = beats_baseline(better.assign(psnr=better.psnr), base, 0.5)
    assert ok and all(d["ok"] for d in det if d["ok"] is not None)
    worse = pd.DataFrame(dict(bpp=[0.15, 0.3], psnr=[20., 22.]))
    assert beats_baseline(worse, base, 0.5)[0] is False
    assert math.isnan(bd_table(better.iloc[:2], {"B": base})["B"])        # < 4 points -> NaN, not a crash


def test_ablation_k_equal_psnr_saving():
    k3 = pd.DataFrame(dict(lmbda=[0.0035, 0.0130, 0.0483], bpp=[0.10, 0.25, 0.60], psnr=[26., 30., 34.]))
    k1 = pd.DataFrame(dict(lmbda=[0.0130], bpp=[0.28], psnr=[30.]))
    ab = ablation_k(k3, k1).iloc[0]
    assert ab.d_bpp_pct_same_lambda == pytest.approx((0.25 / 0.28 - 1) * 100)
    assert ab.saving_pct_equal_psnr == pytest.approx((0.25 / 0.28 - 1) * 100, rel=1e-6)   # K1 sits exactly on a K3 point
