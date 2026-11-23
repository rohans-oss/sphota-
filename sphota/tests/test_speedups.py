"""The speed-ups must not change what is computed: image cache (same crops), fused EMA, channels_last, and the training job graph
(truesight/image_cache.py, truesight/speed.py, truesight/diffusion/train_diffusion.update_ema, scripts/train_phase.py)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from truesight import image_cache, parallel, speed
from truesight.codec.vae_codec import TrueSightCodec
from truesight.data import CropDataset, ROOT
from truesight.diffusion.train_diffusion import update_ema
from truesight.diffusion.unet import make_unet

sys.path.insert(0, str(ROOT / "scripts"))


def _images(tmp_path, n=3, h=300, w=400, seed=1):
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = tmp_path / f"im{i}.png"
        Image.fromarray((rng.random((h + 20 * i, w, 3)) * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join(paths))
    return paths, lst


# ------------------------------------------------------------------------------------------------ image cache
def test_cached_crops_are_exactly_the_crops_of_the_png_path(tmp_path):
    paths, lst = _images(tmp_path)
    out = image_cache.build_cache(paths, tmp_path / "cache", ROOT, workers=1, log=lambda m: None)
    assert out["cached"] == 3 and out["skipped_for_space"] == 0
    plain, cached = CropDataset(lst, 128, cache=False), CropDataset(lst, 128, cache=str(tmp_path / "cache"))
    assert plain.cache is None and cached.cache is not None
    for k in range(30):                                              # the same RNG state -> the same crop position, flip and pixels
        torch.manual_seed(k); a = plain[k % 3]
        torch.manual_seed(k); b = cached[k % 3]
        assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b)


def test_partial_cache_is_still_correct_and_extends_on_the_next_run(tmp_path):
    paths, lst = _images(tmp_path)
    one_image = 300 * 400 * 3
    first = image_cache.build_cache(paths, tmp_path / "c", ROOT, max_bytes=one_image + 10, workers=1, log=lambda m: None)
    assert first["cached"] == 1 and first["skipped_for_space"] == 2           # no room for the others: they stay on the PNG path
    plain, part = CropDataset(lst, 64, cache=False), CropDataset(lst, 64, cache=str(tmp_path / "c"))
    for k in range(9):
        torch.manual_seed(k); a = plain[k % 3]
        torch.manual_seed(k); b = part[k % 3]
        assert torch.equal(a, b)                                              # cached and uncached images give the same tensors
    second = image_cache.build_cache(paths, tmp_path / "c", ROOT, workers=1, log=lambda m: None)   # more room now: extends
    assert second["cached"] == 3 and image_cache.load_index(tmp_path / "c")["bytes"] > one_image


def test_changed_source_is_not_served_from_a_stale_cache(tmp_path):
    paths, lst = _images(tmp_path, n=1)
    image_cache.build_cache(paths, tmp_path / "c", ROOT, workers=1, log=lambda m: None)
    Image.fromarray(np.zeros((350, 450, 3), np.uint8)).save(paths[0])         # the file changed after the cache was built
    c = image_cache.ImageCache(tmp_path / "c", ROOT)
    assert c.get(paths[0]) is None
    assert CropDataset(lst, 64, cache=str(tmp_path / "c"))[0].max() == 0.0     # decoded from the new PNG


def test_cache_survives_an_interrupted_build(tmp_path):
    paths, _ = _images(tmp_path)
    image_cache.build_cache(paths[:2], tmp_path / "c", ROOT, workers=1, log=lambda m: None)
    with open(tmp_path / "c" / image_cache.DATA, "ab") as f:                   # bytes written after the last index (a killed build)
        f.write(b"x" * 1000)
    out = image_cache.build_cache(paths, tmp_path / "c", ROOT, workers=1, log=lambda m: None)
    assert out["cached"] == 3
    c = image_cache.ImageCache(tmp_path / "c", ROOT)
    for p in paths:
        assert np.array_equal(c.get(p), np.asarray(Image.open(p).convert("RGB")))


def test_no_cache_and_off_switch(tmp_path, monkeypatch):
    _, lst = _images(tmp_path, n=1)
    assert CropDataset(lst, 64, cache=str(tmp_path / "missing")).cache is None
    monkeypatch.setenv("TS_IMAGE_CACHE", "off")
    assert CropDataset(lst, 64).cache is None


def test_dataloader_workers_read_the_cache(tmp_path):
    paths, lst = _images(tmp_path)
    image_cache.build_cache(paths, tmp_path / "c", ROOT, workers=1, log=lambda m: None)
    dl = torch.utils.data.DataLoader(CropDataset(lst, 64, cache=str(tmp_path / "c")), batch_size=3, num_workers=2)
    (x,) = list(dl)
    assert x.shape == (3, 3, 64, 64) and 0 <= x.min() and x.max() <= 1


# ------------------------------------------------------------------------------------------------ fused EMA, channels_last
def test_fused_ema_equals_the_plain_loop():
    torch.manual_seed(0)
    net = make_unet(64, (16, 32, 32, 32))
    ema_a, ema_b = make_unet(64, (16, 32, 32, 32)), make_unet(64, (16, 32, 32, 32))
    ema_b.load_state_dict(ema_a.state_dict())
    for _ in range(3):
        update_ema(ema_a, net, 0.9)
        with torch.no_grad():
            for pe, pm in zip(ema_b.parameters(), net.parameters()):
                pe.mul_(0.9).add_(pm.detach(), alpha=0.1)
    for pa, pb in zip(ema_a.parameters(), ema_b.parameters()):
        assert torch.equal(pa, pb)


def test_speed_settings(monkeypatch):
    monkeypatch.delenv("TS_CHANNELS_LAST", raising=False)
    assert speed.channels_last_mode() == "unet"
    assert not speed.use_channels_last("unet", "cpu") and not speed.use_channels_last("codec", "cpu")   # CPU: nothing changes
    monkeypatch.setenv("TS_CHANNELS_LAST", "bogus")
    with pytest.raises(ValueError):
        speed.channels_last_mode()
    monkeypatch.setenv("TS_CUDNN_BENCHMARK", "0")
    assert not speed.cudnn_benchmark_enabled()


def test_codec_gives_the_same_result_in_channels_last_layout():
    """channels_last changes only the memory layout: the rate, the reconstruction and the coder's CPU parameters stay the same."""
    torch.manual_seed(0)
    a = TrueSightCodec(N=16, M=24).eval()
    b = TrueSightCodec(N=16, M=24).eval()
    b.load_state_dict(a.state_dict())
    speed.to_channels_last(b.g_a, b.g_s, b.h_a)
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        oa, ob = a(x), b(speed.cl(x, True))                        # h_s output is channels_last here: gmm_params must still reshape
    assert torch.allclose(oa["x_hat"], ob["x_hat"], atol=1e-4)
    for k in ("y", "z"):
        assert torch.allclose(oa["likelihoods"][k], ob["likelihoods"][k], atol=1e-5)


# ------------------------------------------------------------------------------------------------ the training job graph
def _touch(path, sleep, rc=0):
    """A fake job: record start / end, then create the file the real job would leave (its final.pt / done-marker)."""
    return [sys.executable, "-c", f"import time,pathlib; s=time.time(); time.sleep({sleep}); p=pathlib.Path({str(path)!r}); "
            f"p.parent.mkdir(parents=True, exist_ok=True); p.write_text('%s %s' % (s, time.time()))"]


def test_training_graph_orders_jobs_and_keeps_both_gpus_busy(tmp_path):
    import train_phase as tp
    a = argparse.Namespace(exp=str(tmp_path), codec_dir=str(tmp_path / "c"), base_iters=1, finetune_iters=1, batch=1, crop=64, N=16,
                           M=24, workers=0, ckpt_every=1, train_list=None, from_scratch=False)
    jobs = tp.build_phase_jobs(a, ["ae", "step9", "codec", "codec_ablation", "diffusion", "train_ablations"])
    names = [j.name for j in jobs]
    by = {j.name: j for j in jobs}
    assert len(names) == len(set(names)) == 14
    assert by["stage:diffusion"].after == ["lmbda0.0018"] and by["stage:train_ablations"].after == ["lmbda0.0018"]   # not on all six codecs
    assert by["lmbda0.0130"].after == [] and by["K1_lmbda0.0130"].after == [] and by["relu_lmbda0.0130"].after == []
    assert names.index("K1_lmbda0.0130") < names.index("lmbda0.0018")          # both GPUs start on a base run
    assert by["stage:diffusion"].env["STAGES"] == "diffusion" and by["stage:diffusion"].env["TS_INNER"] == "1"
    assert all(j.log for j in jobs)
    # run the graph with fake commands (the file each job leaves is its skip_if path)
    for j in jobs:
        j.cmd = _touch(j.skip_if, 0.4 if j.name.startswith("lmbda") or j.name.startswith("K1") or j.name.startswith("relu") else 0.2)
        j.log = None
    assert parallel.run_jobs(jobs, gpus=["0", "1"], poll=0.05, log=lambda m: None) == 0
    t = {n: tuple(float(v) for v in Path(by[n].skip_if).read_text().split()) for n in names}
    assert t["stage:diffusion"][0] >= t["lmbda0.0018"][1] and t["lmbda0.0018"][0] >= t["lmbda0.0130"][1]
    assert t["stage:diffusion"][0] < t["lmbda0.0250"][1] + 5 and t["stage:diffusion"][0] < max(v[1] for v in t.values())
    # the 0.0130 base and the K=1 base ran at the same time (one on each GPU)
    assert t["lmbda0.0130"][0] < t["K1_lmbda0.0130"][1] and t["K1_lmbda0.0130"][0] < t["lmbda0.0130"][1]
    for n in names:                                                              # nothing started before what it needs had finished
        for d in by[n].after:
            assert t[n][0] >= t[d][1] - 1e-3


def test_training_graph_without_the_codec_stage_has_no_dangling_dependency(tmp_path):
    import train_phase as tp
    a = argparse.Namespace(exp=str(tmp_path), codec_dir=str(tmp_path / "c"), base_iters=1, finetune_iters=1, batch=1, crop=64, N=16,
                           M=24, workers=0, ckpt_every=1, train_list=None, from_scratch=False)
    jobs = tp.build_phase_jobs(a, ["diffusion", "ae"])
    assert [j.name for j in jobs] == ["stage:diffusion", "stage:ae"] and all(not j.after for j in jobs)
    parallel.validate(jobs)
