import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from truesight.analysis.features import FEATURES, image_features, jpeg_bpp

ROOT = Path(__file__).resolve().parents[1]


def _solid(rgb, size=(64, 64)):
    return Image.new("RGB", size, rgb)


def test_feature_keys():
    assert set(image_features(_solid((10, 20, 30)))) == set(FEATURES)


def test_flat_image_has_no_texture():
    f = image_features(_solid((128, 128, 128)))
    assert f["grad_mean"] == 0 and f["lap_var"] == 0 and f["entropy"] == 0 and f["edge_density"] == 0
    assert f["colorfulness"] == 0                               # gray: no chroma
    assert f["brightness"] == pytest.approx(128, abs=1)


def test_colorfulness_no_uint8_wraparound():
    # R=0, G=100, B=0: rg = -100 (would wrap to +156 in uint8), yb = 50
    f = image_features(_solid((0, 100, 0)))
    assert f["colorfulness"] == pytest.approx(0.3 * np.hypot(100, 50), rel=1e-3)


def test_texture_raises_complexity_features():
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    flat = _solid((100, 100, 100), (128, 128))
    n, f = image_features(noisy), image_features(flat)
    for k in ("grad_mean", "lap_var", "entropy", "edge_density", "jpeg50_bpp"):
        assert n[k] > f[k], k


def test_edge_image_has_edges():
    a = np.zeros((64, 64, 3), np.uint8); a[:, 32:] = 255
    f = image_features(Image.fromarray(a))
    assert f["edge_density"] > 0 and f["grad_mean"] > 0
    assert f["entropy"] == pytest.approx(1.0, abs=1e-6)         # two equally likely gray levels = 1 bit


def test_jpeg_bpp_consistent_and_monotone():
    rng = np.random.default_rng(1)
    im = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert image_features(im)["jpeg50_bpp"] == pytest.approx(jpeg_bpp(im, 50))
    assert jpeg_bpp(im, 10) < jpeg_bpp(im, 50) < jpeg_bpp(im, 90)


def _load_script():
    spec = importlib.util.spec_from_file_location("tsp", ROOT / "scripts" / "train_size_predictor.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_models_recover_linear_signal_and_lasso_sparsity():
    m = _load_script()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 7))
    y = 2 * X[:, 0] - 1 * X[:, 1] + 0.05 * rng.normal(size=400)    # only 2 relevant features
    for kind in ("Linear", "Ridge", "Lasso", "ElasticNet"):
        assert m.make(kind, 0.01).fit(X, y).score(X, y) > 0.98
    coef = m.make("Lasso", 0.1).fit(X, y).named_steps["model"].coef_
    assert np.all(coef[2:] == 0) and coef[0] > 0 > coef[1]        # irrelevant features are exactly zero
    mu, sd = m.nested_cv(X, y, "Lasso")
    assert mu > 0.95 and sd < 0.05
