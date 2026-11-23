"""Second round of speed-ups: tiled cache, GPU-side flip, codec fp16 hybrid plumbing, evaluation reuse (prefix / first step / batched
guidance), memoised detectors, and the parity-gate decision rules. Everything exact must be exactly equal on the CPU."""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from truesight import image_cache, speed
from truesight.codec.train_codec import train_codec
from truesight.codec.vae_codec import TrueSightCodec
from truesight.data import CropDataset, ROOT, to_float_batch
from truesight.diffusion.decode import coarse_decode, diffusion_decode
from truesight.diffusion.sample import FirstStepCache, PrefixCache, divergence_index
from truesight.diffusion.schedule import Schedule
from truesight.diffusion.unet import make_unet
from truesight.protect.detect import ImageMemo

sys.path.insert(0, str(ROOT / "scripts"))
SMALL = (16, 32, 32, 32)


def _png(tmp_path, name, h, w, seed):
    p = tmp_path / name
    Image.fromarray((np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)).save(p)
    return str(p)


# ------------------------------------------------------------------------------------------------ tiled cache
def test_tiled_cache_reads_any_region_exactly_including_edges_and_odd_sizes(tmp_path):
    paths = [_png(tmp_path, "a.png", 300, 410, 1), _png(tmp_path, "b.png", 128, 128, 2), _png(tmp_path, "c.png", 131, 259, 3)]
    out = image_cache.build_cache(paths, tmp_path / "c", ROOT, workers=1, log=lambda m: None)
    assert out["cached"] == 3
    c = image_cache.ImageCache(tmp_path / "c", ROOT)
    rng = np.random.default_rng(0)
    for p in paths:
        ref = np.asarray(Image.open(p).convert("RGB"))
        assert np.array_equal(c.get(p), ref)                                            # the whole image
        h, w = ref.shape[:2]
        for _ in range(40):                                                             # random regions, crossing tile borders
            hh, ww = int(rng.integers(1, min(h, 200) + 1)), int(rng.integers(1, min(w, 200) + 1))
            top, left = int(rng.integers(0, h - hh + 1)), int(rng.integers(0, w - ww + 1))
            assert np.array_equal(c.read_region(p, top, left, hh, ww), ref[top:top + hh, left:left + ww])
    with pytest.raises(ValueError):
        c.read_region(paths[1], 0, 0, 129, 10)


def test_a_crop_touches_few_contiguous_tiles(tmp_path):
    """The point of the layout: a 128 crop is at most 4 tiles, a 256 crop at most 9 (instead of 128 / 256 scattered row reads)."""
    for crop, most in ((128, 4), (256, 9)):
        worst = max(((top + crop - 1) // 128 - top // 128 + 1) * ((left + crop - 1) // 128 - left // 128 + 1)
                    for top in range(0, 600, 7) for left in range(0, 600, 7))
        assert worst <= most


def test_old_cache_layout_is_ignored_and_rebuilt(tmp_path):
    paths = [_png(tmp_path, "a.png", 140, 150, 1)]
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / image_cache.INDEX).write_text(json.dumps({"version": 1, "bytes": 5, "files": {"x": {}}}))   # the first, untiled format
    assert image_cache.load_index(tmp_path / "c") is None
    assert image_cache.build_cache(paths, tmp_path / "c", ROOT, workers=1, log=lambda m: None)["cached"] == 1


# ------------------------------------------------------------------------------------------------ raw loader + GPU flip
def test_raw_crops_are_the_same_crops_as_the_float_path(tmp_path):
    paths = [_png(tmp_path, f"i{i}.png", 200 + 30 * i, 260, i) for i in range(3)]
    lst = tmp_path / "l.txt"; lst.write_text("\n".join(paths))
    image_cache.build_cache(paths, tmp_path / "c", ROOT, workers=1, log=lambda m: None)
    for cache in (False, str(tmp_path / "c")):
        flt, raw = CropDataset(lst, 128, cache=cache), CropDataset(lst, 128, cache=cache, raw=True)
        for k in range(24):
            torch.manual_seed(k); a = flt[k % 3]
            torch.manual_seed(k); r = raw[k % 3]
            assert r.dtype == torch.uint8 and r.shape == (3, 128, 128)
            f = r.float() / 255
            assert torch.equal(a, f) or torch.equal(a, f.flip(-1))                      # the float path only adds the random flip


def test_gpu_side_flip_is_a_fair_coin_per_sample_and_a_float_batch_passes_through():
    x = torch.randint(0, 255, (400, 3, 8, 8), dtype=torch.uint8)
    torch.manual_seed(0)
    y = to_float_batch(x, "cpu")
    flipped = torch.tensor([not torch.equal(y[i], x[i].float() / 255) for i in range(400)])
    assert 0.42 < flipped.float().mean() < 0.58
    assert all(torch.equal(y[i], x[i].float().div(255)) or torch.equal(y[i], x[i].float().div(255).flip(-1)) for i in range(400))
    assert torch.equal(to_float_batch(x, "cpu", flip=False), x.float() / 255)
    f = torch.rand(2, 3, 4, 4)
    assert to_float_batch(f, "cpu") is f


# ------------------------------------------------------------------------------------------------ codec fp16 hybrid plumbing
def test_amp_flag_is_ignored_without_cuda_and_the_default_path_is_unchanged():
    torch.manual_seed(0)
    a, b = TrueSightCodec(N=16, M=24).eval(), TrueSightCodec(N=16, M=24).eval()
    b.load_state_dict(a.state_dict())
    b.amp = True                                                       # no CUDA tensor: _seq falls back to the plain stack
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        oa, ob = a(x), b(x)
    assert torch.equal(oa["x_hat"], ob["x_hat"]) and torch.equal(oa["y"], ob["y"])


def test_codec_training_runs_with_amp_requested_and_resumes(tmp_path):
    lst = tmp_path / "l.txt"
    lst.write_text("\n".join(_png(tmp_path, f"t{i}.png", 128, 128, i) for i in range(3)))
    net = train_codec(0.013, tmp_path / "run", N=16, M=24, iters=6, batch=2, crop=64, workers=0, ckpt_every=3, train_list=str(lst),
                      cache=None, amp=True, device="cpu") if False else train_codec(
        0.013, tmp_path / "run", N=16, M=24, iters=6, batch=2, crop=64, workers=0, ckpt_every=3, train_list=str(lst), amp=True, device="cpu")
    assert (tmp_path / "run" / "final.pt").exists() and net.amp is False and net._amp_skipped == 0


def test_gated_flags(tmp_path, monkeypatch):
    f = tmp_path / "p.json"
    monkeypatch.setenv("TS_X", "auto"); monkeypatch.setenv("TS_F", str(f))
    assert speed.gated_flag("TS_X", "TS_F", "ok") is False                              # no file yet: off
    f.write_text(json.dumps({"ok": False})); assert speed.gated_flag("TS_X", "TS_F", "ok") is False
    f.write_text(json.dumps({"ok": True})); assert speed.gated_flag("TS_X", "TS_F", "ok") is True
    monkeypatch.setenv("TS_X", "0"); assert speed.gated_flag("TS_X", "TS_F", "ok") is False
    monkeypatch.setenv("TS_X", "1"); assert speed.gated_flag("TS_X", "TS_F", "ok") is True
    f.write_text("not json"); monkeypatch.setenv("TS_X", "auto"); assert speed.gated_flag("TS_X", "TS_F", "ok") is False
    assert not speed.codec_amp_enabled("cpu") and not speed.eval_amp_enabled("cpu") and not speed.cond_amp_enabled("cpu")


def test_amp_parity_decision_rule():
    ap = importlib.import_module("amp_parity")
    fp32 = [(0.200, 30.00), (0.201, 30.04)]
    same = ap.verdict(fp32, [(0.2005, 30.02), (0.2008, 30.01)], 1.3, 100, 0)
    assert same["amp_ok"] and same["speedup"] == 1.3
    worse_psnr = ap.verdict(fp32, [(0.2, 29.5), (0.2, 29.6)], 1.3, 100, 0)
    worse_bpp = ap.verdict(fp32, [(0.21, 30.0), (0.211, 30.02)], 1.3, 100, 0)
    skipping = ap.verdict(fp32, [(0.2, 30.0), (0.2, 30.0)], 1.3, 100, 5)
    nonfinite = ap.verdict(fp32, [(0.2, 30.0), (0.2, 30.0)], 1.3, 100, 0, finite=False)
    assert not worse_psnr["amp_ok"] and not worse_bpp["amp_ok"] and not skipping["amp_ok"] and not nonfinite["amp_ok"]


def test_eval_parity_decision_rule():
    ep = importlib.import_module("eval_parity")
    ok = ep.decide([80, 75], [45, 44], [20, 22], 0.002)
    assert ok == {"batch_ok": True, "amp_ok": True}
    assert not ep.decide([80, 50], [45, 44], [20, 22], 0.0)["batch_ok"]                  # batched decode differs too much
    assert not ep.decide([80, 80], [30, 30], [20, 22], 0.0)["amp_ok"]                     # fp16 decode too far from fp32
    assert not ep.decide([80, 80], [38, 38], [35, 35], 0.0)["amp_ok"]                     # not clearly below the seed-to-seed noise
    assert not ep.decide([80, 80], [45, 45], [20, 20], 0.05)["amp_ok"]                    # LPIPS got worse
    assert ep.decide([80], [45], [20], 0.0, cuda=False) == {"batch_ok": False, "amp_ok": False}


# ------------------------------------------------------------------------------------------------ evaluation reuse (exact on CPU)
@pytest.fixture(scope="module")
def tiny_unet():
    torch.manual_seed(0)
    u = make_unet(64, SMALL).eval()
    for p in u.parameters():                                           # a non-trivial network (the default init predicts ~0)
        p.data.add_(0.05 * torch.randn_like(p))
    return u, Schedule(1000, "cosine")


class Guide:
    """Changes x0_hat from t <= t_start on (like the faithfulness guide)."""
    k, alpha, roi_mask = 1, 1.0, None

    def __init__(self, t_start=700):
        self.t_start = t_start

    def __call__(self, x0, t):
        return x0 if t > self.t_start else x0 * 0.98


def test_divergence_index():
    sched = Schedule(1000, "cosine")
    ts = sched.ddim_timesteps(10)
    assert divergence_index(ts, None) is None
    g = Guide(700)
    i = divergence_index(ts, g)
    assert ts[i] <= 700 and (i == 0 or ts[i - 1] > 700)
    g.k = 0
    assert divergence_index(ts, g) is None                                               # measure-only: never diverges
    assert divergence_index(ts, lambda x0, t: x0) == 0                                  # an unknown hook: nothing is shared


def test_prefix_reuse_gives_the_same_decode_as_decoding_from_scratch(tiny_unet):
    unet, sched = tiny_unet
    coarse = torch.rand(1, 3, 64, 64)
    kw = dict(steps=12, w=1.5, seed=3)
    scratch_plain = diffusion_decode(unet, sched, coarse, **kw)
    scratch_g700 = diffusion_decode(unet, sched, coarse, guide=Guide(700), **kw)
    scratch_g500 = diffusion_decode(unet, sched, coarse, guide=Guide(500), **kw)
    pc = PrefixCache()
    p1 = diffusion_decode(unet, sched, coarse, prefix=pc, **kw)                          # records the whole unguided path
    assert max(pc.states) == 11
    g700 = diffusion_decode(unet, sched, coarse, guide=Guide(700), prefix=pc, **kw)      # resumes at its first guided step
    g500 = diffusion_decode(unet, sched, coarse, guide=Guide(500), prefix=pc, **kw)
    p2 = diffusion_decode(unet, sched, coarse, prefix=pc, **kw)                          # an unguided repeat resumes at the last step
    for a, b in ((p1, scratch_plain), (g700, scratch_g700), (g500, scratch_g500), (p2, scratch_plain)):
        assert torch.allclose(a, b, atol=1e-6)
    assert not torch.allclose(scratch_g700, scratch_plain, atol=1e-3)                    # the guide really changes the decode


def test_prefix_recorded_by_a_guided_run_serves_a_later_guided_run(tiny_unet):
    unet, sched = tiny_unet
    coarse = torch.rand(1, 3, 64, 64)
    kw = dict(steps=12, w=1.5, seed=0)
    pc = PrefixCache()
    diffusion_decode(unet, sched, coarse, guide=Guide(500), prefix=pc, **kw)             # records only up to its own first guided step
    ts = sched.ddim_timesteps(12)
    assert max(pc.states) == divergence_index(ts, Guide(500))
    again = diffusion_decode(unet, sched, coarse, guide=Guide(700), prefix=pc, **kw)     # diverges earlier: resumes from a recorded state
    assert torch.allclose(again, diffusion_decode(unet, sched, coarse, guide=Guide(700), **kw), atol=1e-6)


def test_prefix_cache_forgets_when_the_settings_change(tiny_unet):
    unet, sched = tiny_unet
    coarse = torch.rand(1, 3, 64, 64)
    pc = PrefixCache()
    a = diffusion_decode(unet, sched, coarse, steps=8, w=1.5, seed=0, prefix=pc)
    b = diffusion_decode(unet, sched, coarse, steps=8, w=2.0, seed=0, prefix=pc)          # another CFG scale: not the same path
    c = diffusion_decode(unet, sched, coarse, steps=8, w=2.0, seed=1, prefix=pc)          # another seed
    assert torch.allclose(b, diffusion_decode(unet, sched, coarse, steps=8, w=2.0, seed=0), atol=1e-6)
    assert torch.allclose(c, diffusion_decode(unet, sched, coarse, steps=8, w=2.0, seed=1), atol=1e-6)
    assert not torch.allclose(a, b, atol=1e-4)


def test_first_step_is_shared_across_cfg_scales_and_step_counts(tiny_unet):
    unet, sched = tiny_unet
    coarse = torch.rand(1, 3, 64, 64)
    fs, calls = FirstStepCache(), []
    orig = unet.forward

    def counting(*a, **k):
        calls.append(a[0].shape[0])
        return orig(*a, **k)
    unet.forward = counting
    try:
        outs = {}
        for w, steps in ((0.0, 8), (1.0, 8), (1.5, 8), (3.0, 8), (1.5, 6)):              # a CFG sweep, then a step-count sweep
            outs[(w, steps)] = diffusion_decode(unet, sched, coarse, steps=steps, w=w, seed=0, first_step=fs)
        n_shared = len(calls)
        calls.clear()
        for w, steps in ((0.0, 8), (1.0, 8), (1.5, 8), (3.0, 8), (1.5, 6)):
            ref = diffusion_decode(unet, sched, coarse, steps=steps, w=w, seed=0)
            assert torch.allclose(outs[(w, steps)], ref, atol=1e-6)
        assert n_shared < len(calls)                                                     # fewer network calls with the shared first step
    finally:
        unet.forward = orig


def test_batched_guidance_passes_equal_two_separate_passes(tiny_unet):
    unet, sched = tiny_unet
    coarse = torch.rand(1, 3, 64, 64)
    seq = diffusion_decode(unet, sched, coarse, steps=8, w=1.7, seed=2, batch_cfg=False)
    bat = diffusion_decode(unet, sched, coarse, steps=8, w=1.7, seed=2, batch_cfg=True)
    assert torch.allclose(seq, bat, atol=1e-4)          # float noise only (another conv summation order for batch 2)


def test_coarse_decode_default_is_unchanged_on_cpu():
    torch.manual_seed(0)
    codec = TrueSightCodec(N=16, M=24).eval()
    x = torch.rand(1, 3, 64, 64)
    assert torch.equal(coarse_decode(codec, x), coarse_decode(codec, x, amp=False))
    assert torch.equal(coarse_decode(codec, x), codec.g_s(torch.round(codec.g_a(x))).clamp(0, 1))


# ------------------------------------------------------------------------------------------------ memoised models, wait meter
def test_image_memo_returns_the_first_result_and_bounds_memory():
    memo, calls = ImageMemo(size=2), []
    a, b, c = (np.full((4, 4, 3), v, np.uint8) for v in (1, 2, 3))
    run = lambda x: (calls.append(1), x.sum())[1]
    assert memo.get(ImageMemo.key(a), lambda: run(a)) == memo.get(ImageMemo.key(a), lambda: run(a)) and len(calls) == 1
    memo.get(ImageMemo.key(b), lambda: run(b)); memo.get(ImageMemo.key(c), lambda: run(c))
    assert len(memo.d) == 2                                                              # the oldest entry was dropped
    assert ImageMemo.key(a) != ImageMemo.key(np.full((2, 8, 3), 1, np.uint8))             # shape is part of the key


def test_ocr_and_face_models_run_once_per_original(monkeypatch):
    from truesight.protect.faithfulness import EasyOCRAdapter, FaceEmbedder, text_faithfulness

    class FakeReader:
        n = 0

        def readtext(self, a):
            FakeReader.n += 1
            return [([[0, 0], [10, 0], [10, 5], [0, 5]], "ab", 0.9)]

        def recognize(self, a, horizontal_list, free_list):
            return [(None, "ab", 0.9)]

    ocr = EasyOCRAdapter(reader=FakeReader())
    orig = Image.fromarray(np.zeros((20, 20, 3), np.uint8))
    decs = [Image.fromarray(np.full((20, 20, 3), v, np.uint8)) for v in (1, 2, 3)]
    for d in decs:                                                                       # three methods, one original
        assert text_faithfulness(ocr, orig, d)["passed"]
    assert FakeReader.n == 1

    calls = []

    class FakeModel:
        def __call__(self, x):
            calls.append(1)
            return torch.ones(1, 4)

    emb = FaceEmbedder(model=FakeModel())
    for _ in range(3):
        emb.embed(orig)
    assert len(calls) == 1


def test_wait_meter_measures_blocking_time():
    m = speed.WaitMeter()
    with m:
        time.sleep(0.05)
    time.sleep(0.05)
    assert 0.3 < m.fraction() < 0.7
    m.reset()
    assert m.wait == 0.0
