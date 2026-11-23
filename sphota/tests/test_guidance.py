import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from truesight.codec import bitstream as bs
from truesight.codec.entropy_gmm import torchac_available
from truesight.codec.vae_codec import TrueSightCodec
from truesight.diffusion.decode import diffusion_decode
from truesight.diffusion.eval_guidance import (choose_guidance_config, done_criteria, lpips_degradation_pct, mean_history,
                                               safe_ratio, wins_by_cosine, wins_by_pass)
from truesight.diffusion.guidance import FaithfulGuide, consistency_rate, pad_to_size, sphota_decode
from truesight.diffusion.schedule import Schedule
from truesight.diffusion.unet import make_unet
from truesight.protect import roi

needs_torchac = pytest.mark.skipif(not torchac_available(), reason="torchac cannot be imported (no C++ compiler on PATH)")


def _codec(seed=0, scale=25.0):
    """Tiny codec whose latents span several integers (so the rounding bins matter)."""
    torch.manual_seed(seed)
    net = TrueSightCodec(N=16, M=24, K=3).eval()
    with torch.no_grad():
        net.g_a[-1].weight.mul_(scale)
    net.requires_grad_(False)
    return net


# --------------------------------------------------------------------------- losses
def test_bin_loss_is_zero_inside_the_bins_and_matches_the_formula_outside():
    net = _codec()
    x = torch.rand(1, 3, 64, 64)
    y = net.g_a(x)
    assert FaithfulGuide(net, y.detach()).loss(x)[1] == 0.0                              # y_hat == y: exactly inside
    for shift, want in ((1.0, (1.0 - 0.5 + 0.05) ** 2), (0.4, 0.0), (0.46, 0.01 ** 2)):  # margin: 0.5 - 0.05 = 0.45
        lb = FaithfulGuide(net, (y + shift).detach()).loss(x)[1]
        assert lb == pytest.approx(want, abs=1e-6)


def test_roi_loss_is_the_masked_mean_squared_error():
    net = _codec()
    x = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., :, :32] = 1
    g = FaithfulGuide(net, net.g_a(x).detach(), roi_img=x + 0.2, roi_mask=mask)
    _, lb, lro = g.loss(x)
    assert lb == 0.0 and lro == pytest.approx(0.04, abs=1e-6)
    assert FaithfulGuide(net, net.g_a(x).detach(), roi_img=x, roi_mask=torch.zeros(1, 1, 64, 64)).loss(x)[2] == 0.0   # empty mask


def test_total_loss_is_the_specified_weighted_sum_and_normalisation_rescales_it():
    net = _codec()
    x = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., :32, :32] = 1
    y_hat = (net.g_a(x) + 1).detach()
    plain = FaithfulGuide(net, y_hat, x + 0.1, mask, alpha=2.0, beta=10.0, normalize=False)
    tot, lb, lro = plain.loss(x)
    assert tot.item() == pytest.approx(2.0 * lb + 10.0 * lro, rel=1e-5)                 # literal alpha * L_bin + beta * L_roi
    norm = FaithfulGuide(net, y_hat, x + 0.1, mask, alpha=2.0, beta=10.0, normalize=True)
    tot_n, lb_n, lro_n = norm.loss(x)
    assert (lb_n, lro_n) == (lb, lro)                                                    # reported values are the plain means
    s_bin, s_roi = (4 * 4) / 64, 32 * 32                                                 # 16 latent positions; 1024 masked pixels
    assert tot_n.item() == pytest.approx(2.0 * s_bin * lb + 10.0 * s_roi * lro, rel=1e-5)


def _mean_grad(h, w, normalize):
    net = _codec()
    torch.manual_seed(1)
    y_hat = torch.round(net.g_a(torch.rand(1, 3, h, w)))
    g = FaithfulGuide(net, y_hat, normalize=normalize)
    x = (torch.rand(1, 3, h, w) * 2 - 1).requires_grad_(True)
    gr, = torch.autograd.grad(g.loss((x + 1) / 2)[0], x)
    return gr.abs().mean().item()


def test_normalisation_makes_the_step_size_independent_of_the_image_size():
    lit = _mean_grad(128, 128, False) / _mean_grad(256, 256, False)
    nrm = _mean_grad(128, 128, True) / _mean_grad(256, 256, True)
    assert lit > 2.5                                                                     # literal mean loss: gradient ~ 1/(#latents)
    assert 0.5 < nrm < 2.0                                                               # normalised: roughly constant


# --------------------------------------------------------------------------- consistency rate
def test_consistency_rate_counts_latents_inside_their_bins():
    net = _codec()
    x = torch.rand(1, 3, 64, 64)
    y = net.g_a(x)
    assert consistency_rate(net, x, torch.round(y)) == 1.0
    y_hat = y.clone()
    y_hat.view(-1)[::2] += 2.0                                                           # half the elements far outside
    assert consistency_rate(net, x, y_hat) == pytest.approx(0.5)
    big = torch.rand(1, 3, 100, 90)                                                      # padded like the encoder does
    assert consistency_rate(net, big, torch.round(net.g_a(bs.pad_to_multiple(big, 64)))) == 1.0


# --------------------------------------------------------------------------- the guide
def test_guide_skips_noisy_steps_and_records_guided_ones():
    net = _codec()
    x0 = torch.rand(1, 3, 64, 64) * 2 - 1
    g = FaithfulGuide(net, torch.round(net.g_a(torch.rand(1, 3, 64, 64))), lr=5.0, t_start=700)
    assert g(x0, 900) is x0 and g.history == []                                          # t > t_start: untouched
    out = g(x0, 600)
    assert out.shape == x0.shape and not torch.equal(out, x0) and out.min() >= -1 and out.max() <= 1
    (t, lb0, lr0, lb1, lr1), = g.history
    assert t == 600 and lb1 < lb0 and lr0 == lr1 == 0.0                                  # one call lowers L_bin
    assert not out.requires_grad


def test_guide_with_nothing_to_enforce_is_a_no_op_and_zero_inner_steps_only_measures():
    net = _codec()
    x0 = torch.rand(1, 3, 64, 64) * 2 - 1
    y_hat = torch.round(net.g_a(torch.rand(1, 3, 64, 64)))
    nop = FaithfulGuide(net, y_hat, alpha=0.0)
    assert nop(x0, 100) is x0 and nop.history == []
    meas = FaithfulGuide(net, y_hat, inner_steps=0, t_start=999)
    assert torch.equal(meas(x0, 500), x0) and len(meas.history) == 1
    assert meas.history[0][1] == meas.history[0][3] > 0                                  # before == after, and it is measured


def test_one_unnormalised_step_is_the_literal_gradient_step():
    net = _codec()
    torch.manual_seed(2)
    x0 = torch.rand(1, 3, 64, 64) * 2 - 1
    y_hat = torch.round(net.g_a(torch.rand(1, 3, 64, 64)))
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 16:48, 16:48] = 1
    roi_img = torch.rand(1, 3, 64, 64)
    alpha, beta, lr = 1.0, 10.0, 3.0
    g = FaithfulGuide(net, y_hat, roi_img, mask, alpha=alpha, beta=beta, inner_steps=1, lr=lr, normalize=False, max_step=None)
    got = g(x0, 100)
    x = x0.clone().requires_grad_(True)                                                  # the specification, written out by hand
    x01 = (x + 1) / 2
    l_bin = F.relu((net.g_a(x01) - y_hat).abs() - 0.5 + 0.05).pow(2).mean()
    l_roi = (mask * (x01 - roi_img) ** 2).sum() / (3 * mask.sum())
    grad, = torch.autograd.grad(alpha * l_bin + beta * l_roi, x)
    assert torch.allclose(got, (x0 - lr * grad).clamp(-1, 1), atol=1e-6)


class _Trust:
    """Stand-in denoiser that trusts its input (eps = 0, so x0_hat = x_t / sqrt(abar_t)): corrections made to x0_hat persist
    from step to step, as they do with a trained U-Net (a random U-Net would emit fresh junk every step)."""
    def __call__(self, inp, tt):
        return SimpleNamespace(sample=torch.zeros_like(inp[:, :3]))


def _guided_run(lo, y_hat, **kw):
    from truesight.diffusion.sample import ddim_sample_eps
    sched = Schedule(50, "cosine")
    x_T = torch.randn(1, 3, 128, 128, generator=torch.Generator().manual_seed(3))
    g = FaithfulGuide(lo, y_hat, t_start=1000, **kw)
    out = ddim_sample_eps(lambda x, tt: torch.zeros_like(x), sched, x_T.shape, steps=25, x_T=x_T, guide=g)
    return consistency_rate(lo, (out + 1) / 2, y_hat), g


def test_the_step_cap_prevents_divergence_with_an_over_strong_lr():
    lo = _codec()
    torch.manual_seed(5)
    y_hat = torch.round(lo.g_a(torch.rand(1, 3, 128, 128)))
    wild, g_wild = _guided_run(lo, y_hat, lr=1000.0, inner_steps=6, max_step=None)          # literal step: oscillates and blows up
    tame, g_tame = _guided_run(lo, y_hat, lr=1000.0, inner_steps=6)                         # default cap 0.1: converges
    assert wild < 0.2 and g_wild.history[-1][3] > 1.0
    assert tame >= 0.95 and g_tame.history[-1][3] < 0.01


def test_no_pixel_moves_more_than_max_step_per_inner_step():
    lo = _codec()
    torch.manual_seed(5)
    y_hat = torch.round(lo.g_a(torch.rand(1, 3, 64, 64)))
    x0 = torch.rand(1, 3, 64, 64) * 2 - 1
    for cap in (0.02, 0.1):
        out = FaithfulGuide(lo, y_hat, lr=1e4, inner_steps=1, max_step=cap)(x0, 100)
        assert 0 < (out - x0).abs().max().item() <= cap + 1e-6


def test_repeated_guidance_pulls_a_random_image_into_the_transmitted_bins():
    net = _codec()
    torch.manual_seed(1)
    y_hat = torch.round(net.g_a(torch.rand(1, 3, 128, 128)))
    x0 = torch.rand(1, 3, 128, 128) * 2 - 1
    c0 = consistency_rate(net, (x0 + 1) / 2, y_hat)
    g = FaithfulGuide(net, y_hat, inner_steps=3, lr=30.0, t_start=1000)
    for _ in range(60):
        x0 = g(x0, 500)
    assert c0 < 0.5 and consistency_rate(net, (x0 + 1) / 2, y_hat) >= 0.95
    assert g.history[-1][3] < 0.1 * g.history[0][1]                                      # L_bin falls by more than 90%
    assert x0.min() >= -1 and x0.max() <= 1


def test_roi_lock_pulls_the_protected_pixels_to_the_decoded_crops_and_touches_nothing_else():
    net = _codec()
    x0 = torch.rand(1, 3, 64, 64) * 2 - 1
    roi_img = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 8:40, 8:40] = 1
    g = FaithfulGuide(net, torch.zeros(1, 24, 4, 4), roi_img, mask, alpha=0.0, beta=10.0, inner_steps=3, lr=0.05, t_start=1000)
    x = x0
    for _ in range(20):
        x = g(x, 100)
    inside = mask.bool().expand(1, 3, 64, 64)
    err0 = ((x0 + 1) / 2 - roi_img)[inside].abs().mean().item()
    err1 = ((x + 1) / 2 - roi_img)[inside].abs().mean().item()
    # each inner step removes lr * beta / 6 = 8.3% of the remaining error (x is in [-1, 1], the loss in [0, 1]): 60 steps
    assert err1 == pytest.approx(err0 * (1 - 0.05 * 10 / 6) ** 60, rel=0.1) and err1 < 0.01 * err0
    assert torch.equal(x[~inside], x0[~inside])                                          # exactly unchanged outside the boxes


# --------------------------------------------------------------------------- sphota_decode (real files, torchac)
def _file(lo, hi, boxes):
    torch.manual_seed(5)
    x = torch.rand(1, 3, 128, 128)
    blob, _ = roi.encode_protected(lo, hi, x, boxes)
    return x, blob


@needs_torchac
def test_sphota_decode_variants_and_ablation_switches():
    lo, hi = _codec(0), _codec(1, scale=1.0)
    unet = _Trust()                                                                      # corrections persist, like a trained U-Net
    sched = Schedule(50, "cosine")
    boxes = [(32, 32, 64, 64)]
    x, blob = _file(lo, hi, boxes)
    prot = roi.decode_protected(lo, hi, blob)
    kw = dict(prot=prot, w=1.0, steps=10, seed=3)

    plain = sphota_decode(lo, hi, unet, sched, guided=False, use_roi=False, **kw)
    ref = diffusion_decode(unet, sched, prot.coarse, steps=10, w=1.0, seed=3)
    assert torch.equal(plain.x, ref.clamp(0, 1)) and plain.guide is None and plain.boxes == []          # plain = no guide, no ROI

    only_roi = sphota_decode(lo, hi, unet, sched, guided=False, use_roi=True, **kw)
    hard, soft = prot.mask.bool(), prot.soft_mask
    assert torch.equal(only_roi.x_diffusion, plain.x_diffusion) and only_roi.boxes == boxes
    full = soft == 1
    assert torch.allclose(only_roi.x.expand(1, 3, 128, 128)[full.expand(1, 3, 128, 128)],
                          prot.canvas[full.expand(1, 3, 128, 128)], atol=1e-6)                       # protected core = decoded crops
    assert torch.equal(only_roi.x[(soft == 0).expand(1, 3, 128, 128)], plain.x[(soft == 0).expand(1, 3, 128, 128)])   # outside: diffusion

    gk = dict(lr=1000.0, inner_steps=6)
    no_roi = sphota_decode(lo, hi, unet, sched, guided=True, use_roi=False, **gk, **kw)
    assert no_roi.boxes == [] and torch.equal(no_roi.x, no_roi.x_diffusion.clamp(0, 1)) and len(no_roi.guide.history) > 0
    assert no_roi.guide.roi_mask is None                                                                 # ROI-lock is off with the boxes

    both = sphota_decode(lo, hi, unet, sched, guided=True, use_roi=True, **gk, **kw)
    assert both.guide.roi_mask is not None and both.guide.roi_mask.shape[-2:] == (128, 128) and both.boxes == boxes
    c_plain = consistency_rate(lo, plain.x, prot.y_hat)
    c_guided = consistency_rate(lo, no_roi.x, prot.y_hat)
    assert c_plain < 0.5 and c_guided >= 0.95                                                            # guidance restores consistency (A4)
    assert both.guide.history[-1][3] < 0.1 * both.guide.history[0][1]                                    # L_bin falls along the steps
    assert not torch.equal(both.x, only_roi.x)


@needs_torchac
def test_track_only_measures_without_changing_the_decode_and_tiling_rejects_guides():
    lo, hi = _codec(0), _codec(1, scale=1.0)
    unet = make_unet(64, (16, 32, 32, 32)).eval()                                        # a real (random) U-Net also works here
    sched = Schedule(50, "cosine")
    _, blob = _file(lo, hi, [])
    prot = roi.decode_protected(lo, hi, blob)
    base = sphota_decode(lo, hi, unet, sched, prot=prot, guided=False, use_roi=False, steps=5, seed=1)
    meas = sphota_decode(lo, hi, unet, sched, prot=prot, guided=False, use_roi=False, steps=5, seed=1, track_only=True)
    assert torch.equal(base.x, meas.x) and len(meas.guide.history) == 5                  # one record per DDIM step
    assert all(h[1] == h[3] for h in meas.guide.history)
    with pytest.raises(NotImplementedError):
        sphota_decode(lo, hi, unet, sched, prot=prot, guided=True, steps=2, tile=64)


def test_pad_to_size_pads_bottom_and_right_with_zeros():
    t = pad_to_size(torch.ones(1, 1, 2, 3), 4, 5)
    assert t.shape == (1, 1, 4, 5) and t.sum() == 6 and t[..., 2:, :].sum() == 0 and t[..., :, 3:].sum() == 0


# --------------------------------------------------------------------------- tuning / criteria helpers
def test_choose_guidance_config_respects_the_lpips_slack_and_breaks_ties_gently():
    grid = pd.DataFrame(dict(lr=[0.02, 0.05, 0.1, 0.1], t_start=[500, 700, 900, 700],
                             consistency=[0.80, 0.90, 0.99, 0.90], lpips=[0.300, 0.310, 0.400, 0.305]))
    cfg, why = choose_guidance_config(grid, baseline_lpips=0.30, slack=0.10)       # limit 0.33: the 0.99 config is too blurry
    assert (cfg["lr"], cfg["t_start"]) == (0.05, 700) and "highest consistency" in why   # tie at 0.90: the smaller lr
    cfg, why = choose_guidance_config(grid, baseline_lpips=0.20, slack=0.10)       # nothing within 10% -> lowest LPIPS
    assert (cfg["lr"], cfg["t_start"]) == (0.02, 500) and "lowest LPIPS" in why


def test_ratio_and_degradation_helpers():
    assert safe_ratio(60, 30) == 2.0 and safe_ratio(5, 0) == math.inf and math.isnan(safe_ratio(0, 0)) and math.isnan(safe_ratio(float("nan"), 3))
    assert lpips_degradation_pct(0.33, 0.30) == pytest.approx(10.0)


def test_wins_are_counted_only_where_plain_failed_and_sphota_passed():
    assert wins_by_pass([False, True, False, None, False], [True, True, False, True, True]) == [0, 4]
    assert wins_by_cosine([0.4, 0.9, 0.69, 0.3], [0.8, 0.95, 0.7, 0.5]) == [0, 2]


def test_done_criteria_table():
    table = {"plain": dict(consistency=0.4, tfr=20.0, fir=30.0, lpips=0.30),
             "roi_only": dict(consistency=0.5, tfr=70.0, fir=80.0, lpips=0.30),
             "sphota": dict(consistency=0.97, tfr=60.0, fir=50.0, lpips=0.32)}
    c = done_criteria(table, n_examples=7)
    assert c["consistency"]["met"] and c["tfr_factor"]["met"] and c["tfr_factor"]["factor"] == 3.0
    assert c["fir_factor"]["met"] and c["lpips_cost_pct"]["met"] and c["lpips_cost_pct"]["reference"] == "roi_only"
    assert c["visual_examples"]["met"]
    bad = done_criteria({"plain": dict(consistency=0.4, tfr=40.0, fir=40.0, lpips=0.30),
                         "sphota": dict(consistency=0.9, tfr=60.0, fir=50.0, lpips=0.40)}, n_examples=3)
    assert not any(bad[k]["met"] for k in ("consistency", "tfr_factor", "fir_factor", "lpips_cost_pct", "visual_examples"))
    zero = done_criteria({"plain": dict(tfr=0.0, fir=0.0), "sphota": dict(tfr=10.0, fir=5.0)})
    assert zero["tfr_factor"]["factor"] == math.inf and zero["tfr_factor"]["met"]


def test_mean_history():
    h1 = [(900, 1, 0, 0.5, 0), (800, 2, 0, 1.0, 0)]
    h2 = [(900, 3, 0, 1.5, 0), (800, 4, 0, 2.0, 0), (700, 5, 0, 3.0, 0)]
    m = mean_history([h1, h2, []])
    assert m.shape == (2, 5) and m[0, 1] == 2.0 and m[1, 3] == 1.5 and mean_history([]).shape == (0, 5)
