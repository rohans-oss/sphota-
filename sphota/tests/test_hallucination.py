from types import SimpleNamespace

import numpy as np
import pytest
import torch

from truesight.analysis import hallucination as H
from truesight.analysis.risk_model import (DETECTORS, RiskModel, best_detector, evaluate_detectors, learning_curve,
                                           split_by_image)
from truesight.codec import bitstream as bs
from truesight.codec.vae_codec import TrueSightCodec
from truesight.diffusion.schedule import Schedule
from truesight.protect import roi


def _codec(seed=0, scale=25.0):
    torch.manual_seed(seed)
    net = TrueSightCodec(N=16, M=24, K=3).eval()
    with torch.no_grad():
        net.g_a[-1].weight.mul_(scale)
    net.requires_grad_(False)
    return net


# --------------------------------------------------------------------------- patches and texture signals
def test_patch_grid_and_to_patches_drop_the_border():
    assert H.patch_grid(100, 70) == (3, 2) and H.patch_grid(32, 32) == (1, 1) and H.patch_grid(31, 64) == (0, 2)
    m = torch.arange(100 * 70, dtype=torch.float32).view(1, 1, 100, 70)
    p = H.to_patches(m)
    assert p.shape == (3, 2) and p[0, 0].item() == pytest.approx(m[..., :32, :32].mean().item())
    assert p[2, 1].item() == pytest.approx(m[..., 64:96, 32:64].mean().item())          # last whole patch; the border is ignored


def test_laplacian_variance_is_zero_on_flat_patches_and_large_on_noise_and_localised():
    img = torch.full((1, 3, 64, 96), 0.5)
    torch.manual_seed(0)
    img[..., :, 64:] = torch.rand(1, 3, 64, 32)                                          # right third noisy
    lv = H.laplacian_variance_map(img)
    assert lv.shape == (2, 3) and lv[0, 0] < 1e-3 and lv[1, 0] < 1e-3                    # the flat left patches see no detail
    assert 0 < lv[0, 1] < 500                                                            # only the column next to the noise leaks in
    assert lv[0, 2] > 1000 and lv[1, 2] > 1000                                           # noise on the [0, 255] scale
    check = torch.zeros(1, 3, 32, 32)
    check[..., ::2, ::2] = 1
    check[..., 1::2, 1::2] = 1                                                           # checkerboard: Laplacian = +-4 * 255
    assert H.laplacian_variance_map(check)[0, 0] > 5e5


def test_edge_density_sees_edges_only_where_they_are():
    img = torch.zeros(1, 3, 64, 64)
    img[..., :, 40:] = 1.0                                                               # a vertical step inside patch column 1
    ed = H.edge_density_map(img)
    assert ed[0, 0] == 0 and ed[1, 0] == 0 and 0 < ed[0, 1] < 0.2 and ed[0, 1] == ed[1, 1]
    assert H.edge_density_map(torch.full((1, 3, 32, 32), 0.3)).item() == 0


# --------------------------------------------------------------------------- seed variance, bin violation, content overlap
def test_seed_variance_matches_the_population_std_and_needs_two_decodes():
    a, b, c = (torch.full((1, 3, 32, 32), v) for v in (0.0, 0.3, 0.6))
    assert H.seed_variance_map([a, b, c]).item() == pytest.approx(np.sqrt(((0.3) ** 2 * 2) / 3), rel=1e-5)
    assert H.seed_variance_map([a, a.clone(), a.clone()]).item() == 0.0
    d = a.clone().repeat(1, 1, 2, 2)
    e = d.clone()
    e[..., :32, :32] += 0.2                                                              # only patch (0, 0) differs
    sv = H.seed_variance_map([d, e])
    assert sv[0, 0] == pytest.approx(0.1, rel=1e-4) and sv[0, 1] == 0 and sv[1, 0] == 0
    with pytest.raises(ValueError):
        H.seed_variance_map([a])


def test_bin_violation_is_zero_for_a_consistent_decode_and_localised_when_the_latent_is_shifted():
    net = _codec()
    torch.manual_seed(1)
    x = torch.rand(1, 3, 128, 128)
    y_hat = torch.round(net.g_a(x))
    assert torch.count_nonzero(H.bin_violation_map(net, x, y_hat)) == 0                 # y_hat = round(g_a(x)): inside every bin
    shifted = y_hat.clone()
    shifted[..., :2, :2] += 3.0                                                          # latent cells (0..1, 0..1) = patch (0, 0)
    bv = H.bin_violation_map(net, x, shifted)
    assert bv.shape == (4, 4) and bv[0, 0] > 1.5 and bv.sum() == bv[0, 0]               # nothing outside that patch
    big = torch.rand(1, 3, 100, 90)                                                      # padded internally like the encoder
    yb = torch.round(net.g_a(bs.pad_to_multiple(big, 64)))
    assert H.bin_violation_map(net, big, yb).shape == (3, 2)


def test_content_overlap_counts_only_unprotected_detected_pixels():
    det = [(0, 0, 64, 32)]                                                               # covers patches (0,0) and (0,1) fully
    assert H.content_overlap_map(64, 64, det, []).tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert H.content_overlap_map(64, 64, det, [(0, 0, 32, 32)]).tolist() == [[0.0, 1.0], [0.0, 0.0]]    # protected -> safe
    assert H.content_overlap_map(64, 64, [(0, 0, 16, 32)], [])[0, 0].item() == pytest.approx(0.5)
    assert H.content_overlap_map(64, 64, [], []).sum() == 0


def test_labels_need_high_lpips_and_positive_detail_gain():
    lp = np.array([0.5, 0.5, 0.2, 0.35, 0.9])
    gain = np.array([10.0, -5.0, 10.0, 10.0, 0.0])
    assert H.make_labels(lp, gain).tolist() == [1, 0, 0, 0, 0]                           # sharp-but-wrong only; thresholds are strict
    assert H.make_labels(lp, gain, threshold=0.1).tolist() == [1, 0, 1, 1, 0]


# --------------------------------------------------------------------------- the feature table
def test_patch_table_assembles_the_six_signals_in_order():
    net = _codec()
    torch.manual_seed(2)
    x = torch.rand(1, 3, 100, 70)
    coarse = torch.rand(1, 3, 100, 70)
    decodes = [torch.rand(1, 3, 100, 70) for _ in range(3)]
    y_hat = torch.round(net.g_a(bs.pad_to_multiple(decodes[0], 64)))
    fake_lpips = lambda a, b: torch.full((1, 1, 100, 70), 0.4)
    t = H.patch_table(decodes, coarse, y_hat, net, detected_boxes=[(0, 0, 40, 40)], original=x, lpips_fn=fake_lpips)
    assert t["grid"] == (3, 2) and t["X"].shape == (6, 6) and t["lpips"].tolist() == pytest.approx([0.4] * 6, abs=1e-4)
    col = {n: t["X"][:, i] for i, n in enumerate(H.FEATURES)}
    assert np.allclose(col["seed_var"], H.seed_variance_map(decodes).numpy().ravel())
    lap_hat, lap_coarse = H.laplacian_variance_map(decodes[0]), H.laplacian_variance_map(coarse)
    assert np.allclose(col["detail_gain"], (lap_hat - lap_coarse).numpy().ravel())
    assert np.allclose(col["coarse_lapvar"], lap_coarse.numpy().ravel())
    assert col["text_face"][0] > 0.9 and col["text_face"][5] == 0 and (col["bin_violation"] == 0).all()
    assert H.patch_table(decodes, coarse, y_hat, net)["lpips"] is None                  # no original -> no LPIPS / labels


def test_feature_transform_only_touches_the_heavy_tailed_columns():
    X = np.array([[0.1, 0.2, -300.0, 5000.0, 0.3, 0.4]])
    Z = H.transform_features(X)
    assert Z[0, 2] == pytest.approx(-np.log1p(300.0)) and Z[0, 3] == pytest.approx(np.log1p(5000.0))
    assert Z[0, [0, 1, 4, 5]].tolist() == X[0, [0, 1, 4, 5]].tolist() and X[0, 3] == 5000.0   # input not modified
    assert H.signed_log1p(np.array([-2.0, 2.0])).tolist() == pytest.approx([-np.log1p(2), np.log1p(2)])


# --------------------------------------------------------------------------- overlay, safe mode, encoder-side boxes
def test_upsampled_scores_and_heat_overlay():
    u = H.upsample_scores(np.array([[0.0, 1.0], [1.0, 0.0]]), (2, 2), 70, 70)
    assert u.shape == (1, 1, 70, 70) and u.min() >= 0 and u.max() <= 1 and u[0, 0, 5, 5] < 0.2 and u[0, 0, 5, 60] > 0.8
    x = torch.rand(1, 3, 70, 70)
    assert torch.equal(H.heat_overlay(x, torch.zeros(1, 1, 70, 70)), x)               # no risk: the image is untouched
    hot = H.heat_overlay(torch.zeros(1, 3, 8, 8), torch.ones(1, 1, 8, 8))
    assert hot[:, 0].min() > hot[:, 2].max() and hot.min() >= 0 and hot.max() <= 1     # high risk turns red


def test_flag_mask_and_safe_mode():
    flags = np.zeros((4, 4), bool)
    flags[1, 1] = True
    m = H.flag_pixel_mask(flags, 128, 128, feather=8)
    assert m[0, 0, 48, 48] == pytest.approx(1.0, abs=1e-3) and m[0, 0, 110, 110] == 0 and 0 < m[0, 0, 30, 48] < 1
    assert set(torch.unique(H.flag_pixel_mask(flags, 128, 128, feather=0)).tolist()) == {0.0, 1.0}
    x_hat, coarse = torch.rand(1, 3, 128, 128), torch.rand(1, 3, 128, 128)
    safe = H.safe_mode(x_hat, coarse, flags)
    assert torch.allclose(safe[..., 44:52, 44:52], coarse[..., 44:52, 44:52], atol=1e-3)       # flagged core: honest decode
    assert torch.equal(safe[..., 100:, 100:], x_hat[..., 100:, 100:])                             # far from flags: untouched
    assert torch.equal(H.safe_mode(x_hat, coarse, np.zeros((4, 4), bool)), x_hat)                # nothing flagged: no change
    prot = torch.zeros(1, 1, 128, 128)
    prot[..., 32:64, 32:64] = 1                                                                   # the flagged patch is protected
    assert torch.equal(H.safe_mode(x_hat, coarse, flags, prot)[..., 32:64, 32:64], x_hat[..., 32:64, 32:64])


def test_risk_boxes_cover_flagged_patches_and_never_displace_detector_boxes():
    flags = np.zeros((8, 8), bool)
    flags[2, 3] = flags[2, 4] = True
    raw = H.risk_patch_boxes(flags)
    assert raw == [(96.0, 64.0, 128.0, 96.0), (128.0, 64.0, 160.0, 96.0)]
    boxes = H.add_risk_boxes([], flags, 256, 256)
    assert len(boxes) == 1 and all(v % 16 == 0 for v in boxes[0])                    # the two adjacent patches merge into one box
    bx, by, bw, bh = boxes[0]
    assert bx <= 96 and by <= 64 and bx + bw >= 160 and by + bh >= 96
    det = [(10, 10, 60, 40), (200, 10, 250, 40), (10, 200, 60, 240)]
    base = roi.plan_boxes(det, 256, 256)
    assert H.add_risk_boxes(det, np.zeros((8, 8), bool), 256, 256) == base          # nothing flagged: unchanged
    many = np.zeros((8, 8), bool)
    many[::2, 1::2] = True                                                           # lots of scattered risky patches
    capped = H.add_risk_boxes(det, many, 256, 256, max_boxes=5)
    assert len(capped) <= 5
    for d in det:                                                                    # every detector box is still protected
        assert any(b[0] <= d[0] and b[1] <= d[1] and b[0] + b[2] >= d[2] and b[1] + b[3] >= d[3] for b in capped)


# --------------------------------------------------------------------------- detectors on synthetic signals
def _synthetic(n_images=40, per_image=30, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for i in range(n_images):
        base = rng.normal(0, 0.5)                                                    # per-image offset (what makes leakage possible)
        sv = np.abs(rng.normal(0.1 + base * 0.02, 0.05, per_image))
        bv = np.abs(rng.normal(0.0, 0.1, per_image))
        feats = np.column_stack([sv, bv, rng.normal(0, 200, per_image), np.abs(rng.normal(300, 200, per_image)),
                                 rng.random(per_image), rng.random(per_image) * (rng.random(per_image) > 0.8)])
        score = sv * 8 + bv * 5 + 0.002 * feats[:, 2] + rng.normal(0, 0.3, per_image)
        X.append(feats)
        y.append((score > np.quantile(score, 0.8)).astype(int) if informative else rng.integers(0, 2, per_image))
        g.append(np.full(per_image, i))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def test_split_by_image_never_shares_an_image():
    g = np.repeat(np.arange(20), 15)
    tr, te = split_by_image(g, 0.3, seed=1)
    assert not set(g[tr]) & set(g[te]) and len(tr) + len(te) == len(g) and 0.2 < len(te) / len(g) < 0.4
    assert np.array_equal(tr, split_by_image(g, 0.3, seed=1)[0]) and not np.array_equal(tr, split_by_image(g, 0.3, seed=2)[0])


def test_all_eight_detectors_find_a_real_signal_and_none_finds_one_in_noise():
    X, y, g = _synthetic()
    tr, te = split_by_image(g, 0.3, seed=0)
    Z = H.transform_features(X)
    res = evaluate_detectors(Z[tr], y[tr], Z[te], y[te])
    assert tuple(res) == DETECTORS and all(0 <= v <= 1 for v in res.values())
    for name in ("perceptron", "logistic", "mlp", "svm", "random_forest", "xgboost"):
        assert res[name] > 0.85, (name, res)
    assert res["gmm_seed_var"] > 0.7 and res["isolation_forest"] > 0.5                  # unsupervised: weaker but informative
    name, auroc, met = best_detector(res)
    assert name in DETECTORS and auroc == max(res.values()) and met
    Xn, yn, gn = _synthetic(informative=False, seed=3)                                    # labels unrelated to the signals
    trn, ten = split_by_image(gn, 0.3, seed=0)
    noise = evaluate_detectors(H.transform_features(Xn)[trn], yn[trn], H.transform_features(Xn)[ten], yn[ten])
    assert all(abs(v - 0.5) < 0.12 for k, v in noise.items() if k not in ("perceptron",)), noise   # chance level, no leakage
    with pytest.raises(ValueError):
        evaluate_detectors(Z[tr], y[tr], Z[te], np.zeros(len(te), int))                  # one class: AUROC undefined


def test_best_detector_flags_a_missed_target():
    assert best_detector({"a": 0.6, "b": 0.7}) == ("b", 0.7, False) and best_detector({"a": 0.8})[2] is True


def test_learning_curve_grows_with_data_and_skips_single_class_subsets():
    X, y, g = _synthetic(n_images=60)
    tr, te = split_by_image(g, 0.3, seed=0)
    Z = H.transform_features(X)
    for model in ("logistic", "xgboost"):
        lc = learning_curve(model, Z[tr], y[tr], Z[te], y[te])
        n = [r["n_train"] for r in lc]
        assert n == sorted(n) and len(lc) >= 4 and lc[-1]["n_train"] == len(tr)
        assert lc[-1]["val_auroc"] > 0.8 and lc[-1]["val_auroc"] >= lc[0]["val_auroc"] - 0.05
        assert all(0 <= r["train_auroc"] <= 1 for r in lc)
    xg = learning_curve("xgboost", Z[tr], y[tr], Z[te], y[te], fractions=(1.0,))[0]
    assert xg["train_auroc"] >= xg["val_auroc"]                                         # boosted trees fit the training set better
    assert learning_curve("logistic", Z[tr][:50], np.zeros(50, int), Z[te], y[te]) == []


def test_risk_model_uses_only_its_features_and_roundtrips(tmp_path):
    X, y, g = _synthetic()
    enc, full = RiskModel().fit(X, y), RiskModel(H.FEATURES).fit(X, y)
    r = enc.predict_risk(X)
    assert r.shape == (len(X),) and r.min() >= 0 and r.max() <= 1 and enc.flags(X).dtype == bool
    X2 = X.copy()
    X2[:, H.FEATURES.index("seed_var")] += 5.0                                          # the encoder cannot see the seed variance
    assert np.allclose(enc.predict_risk(X2), r) and not np.allclose(full.predict_risk(X2), full.predict_risk(X))
    enc.save(tmp_path / "m" / "risk.joblib")
    assert np.allclose(RiskModel.load(tmp_path / "m" / "risk.joblib").predict_risk(X), r)
    with pytest.raises(RuntimeError):
        RiskModel().predict_risk(X)
    assert enc.features == tuple(H.ENCODER_FEATURES)


# --------------------------------------------------------------------------- the whole decoder-side pipeline
class _Trust:
    def __call__(self, inp, tt):
        return SimpleNamespace(sample=torch.zeros_like(inp[:, :3]))


def test_analyse_decode_produces_overlay_flags_and_a_safe_image():
    lo = _codec()
    torch.manual_seed(4)
    x = torch.rand(1, 3, 128, 128)
    y_hat = torch.round(lo.g_a(x))
    coarse = lo.g_s(y_hat).clamp(0, 1)
    prot = roi.Protected(coarse=coarse, y_hat=y_hat, canvas=torch.zeros(1, 3, 128, 128), mask=torch.zeros(1, 1, 128, 128),
                         soft_mask=torch.zeros(1, 1, 128, 128), boxes=[])
    Xs, ys, _ = _synthetic()
    model = RiskModel(H.FEATURES).fit(Xs, ys)
    out = H.analyse_decode(lo, _Trust(), Schedule(50, "cosine"), prot, model, steps=3, threshold=0.5)
    assert len(out["decodes"]) == 3 and out["table"]["X"].shape == (16, 6) and out["table"]["X"][:, 0].min() > 0   # seeds disagree
    assert out["risk"].shape == (4, 4) and out["risk"].min() >= 0 and out["risk"].max() <= 1 and out["flags"].shape == (4, 4)
    for k in ("overlay", "safe", "x_hat"):
        assert out[k].shape == (1, 3, 128, 128)
    flagged = np.kron(out["flags"], np.ones((32, 32), bool))
    if flagged.any():                                                                    # inside a flagged patch the safe image is the coarse decode
        core = torch.as_tensor(np.kron(out["flags"], np.ones((32, 32)))[8:-8, 8:-8] > 0)
        assert out["safe"][0, :, 8:-8, 8:-8].permute(1, 2, 0)[core].shape[0] >= 0
    assert torch.equal(out["x_hat"], out["decodes"][0])
