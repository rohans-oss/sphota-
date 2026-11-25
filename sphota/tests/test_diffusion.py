from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from truesight.codec.train_codec import model_config
from truesight.codec.vae_codec import TrueSightCodec
from truesight.diffusion.decode import coarse_decode, diffusion_decode, tile_starts
from truesight.diffusion.eval_diffusion import (LPIPS_TARGET_PCT, checks, choose_default_w, fmt_table,
                                                improvement_pct, summarize)
from truesight.diffusion.schedule import Schedule
from truesight.diffusion.train_diffusion import (drop_condition, load_unet, lr_at, train_diffusion, update_ema)
from truesight.diffusion.unet import DOWN, UP, make_unet, norm_groups

SMALL = (16, 32, 32, 32)


# --------------------------------------------------------------------------- U-Net
def test_unet_takes_6_channels_returns_3_and_is_fully_convolutional():
    u = make_unet(128, SMALL)
    assert u.config.in_channels == 6 and u.config.out_channels == 3
    for shape in [(2, 6, 128, 128), (1, 6, 64, 192)]:
        x = torch.randn(*shape)
        assert u(x, torch.randint(0, 1000, (x.shape[0],))).sample.shape == (shape[0], 3, *shape[2:])


def test_unet_has_attention_at_the_middle_resolutions_only():
    assert [d for d in DOWN].index("AttnDownBlock2D") == 2 and "AttnUpBlock2D" in UP
    assert DOWN[0] == "DownBlock2D" and DOWN[-1] == "DownBlock2D" and UP[0] == "UpBlock2D" and UP[-1] == "UpBlock2D"
    u = make_unet(128, SMALL)
    assert any("Attn" in type(b).__name__ for b in u.down_blocks) and u.mid_block.attentions is not None


def test_guide_sized_unet_is_about_25m_parameters_and_group_norm_choice():
    u = make_unet(128)
    assert 20e6 < sum(p.numel() for p in u.parameters()) < 30e6
    assert norm_groups((64, 128, 256, 256)) == 32 and norm_groups((16, 32, 32, 32)) == 16 and norm_groups((12, 24, 24, 24)) == 4
    with pytest.raises(ValueError):
        make_unet(128, (64, 128, 256))


# --------------------------------------------------------------------------- training pieces
def test_lr_schedule_warms_up_then_decays_with_cosine():
    base, total, warm = 1e-4, 1000, 100
    assert lr_at(0, total, warm, base) == 0.0
    assert lr_at(50, total, warm, base) < lr_at(100, total, warm, base)
    assert lr_at(100, total, warm, base) == pytest.approx(base * 0.5 * (1 + np.cos(np.pi * 0.1)), rel=1e-6)
    later = [lr_at(s, total, warm, base) for s in range(100, 1001, 100)]
    assert later == sorted(later, reverse=True) and later[-1] == pytest.approx(0.0, abs=1e-12)
    assert lr_at(10, total, 0, base) > 0                                     # warmup = 0 disables the ramp


def test_drop_condition_zeroes_about_p_of_the_samples_and_leaves_the_rest_intact():
    c = torch.randn(20_000, 3, 2, 2) + 5
    g = torch.Generator().manual_seed(0)
    d = drop_condition(c, 0.1, g)
    dropped = (d.abs().sum((1, 2, 3)) == 0)
    assert dropped.float().mean().item() == pytest.approx(0.1, abs=0.01)
    assert torch.equal(d[~dropped], c[~dropped])
    assert torch.count_nonzero(drop_condition(c, 0.0)) == c.numel() and torch.count_nonzero(drop_condition(c, 1.0)) == 0


def test_ema_update_formula():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    old = [p.clone() for p in a.parameters()]
    update_ema(a, b, 0.9)
    for pa, po, pb in zip(a.parameters(), old, b.parameters()):
        assert torch.allclose(pa, 0.9 * po + 0.1 * pb, atol=1e-6)


def _images(tmp_path, n=4, size=96):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        a = np.kron(rng.random((size // 12, size // 12, 3)), np.ones((12, 12, 1)))
        p = tmp_path / f"i{i}.png"
        Image.fromarray((a * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join(paths))
    return str(lst)


def _codec_ckpt(tmp_path, N=16, M=24):
    torch.manual_seed(0)
    net = TrueSightCodec(N=N, M=M, K=3)
    p = tmp_path / "codec.pt"
    torch.save({"net": net.state_dict(), "it": 0, "config": model_config(net, 0.0018)}, p)
    return str(p), net.eval()


def test_coarse_decode_is_the_rounded_latent_reconstruction():
    net = TrueSightCodec(N=16, M=24, K=3).eval()
    x = torch.rand(1, 3, 128, 64)
    c = coarse_decode(net, x)
    ref = net(x)["x_hat"].clamp(0, 1)                                        # eval forward decodes round(y)
    assert c.shape == x.shape and c.min() >= 0 and c.max() <= 1 and torch.allclose(c, ref, atol=1e-6)
    assert not c.requires_grad


def test_training_resume_final_checkpoint_and_frozen_codec(tmp_path):
    lst = _images(tmp_path)
    codec_path, _ = _codec_ckpt(tmp_path)
    run = tmp_path / "run"
    kw = dict(codec_path=codec_path, out=run, batch=2, crop=64, channels=SMALL, T=100, warmup=2, ckpt_every=3,
              log_every=2, workers=0, train_list=lst, device="cpu")
    train_diffusion(iters=6, stop_at=3, **kw)                                # "killed" at 3
    assert (run / "last.pt").exists() and not (run / "final.pt").exists()
    ema = train_diffusion(iters=6, **kw)                                     # restart -> finishes
    assert (run / "final.pt").exists()
    log = pd.read_csv(run / "log.csv")
    assert log["step"].tolist() == [2, 4, 6] and np.isfinite(log[["loss", "lr"]].to_numpy()).all()
    unet, cfg = load_unet(run / "final.pt")
    assert cfg["T"] == 100 and cfg["kind"] == "cosine" and cfg["codec_lmbda"] == 0.0018 and not unet.training
    assert all(torch.equal(a, b) for a, b in zip(unet.state_dict().values(), ema.state_dict().values()))
    raw, _ = load_unet(run / "final.pt", use_ema=False)
    assert any(not torch.equal(a, b) for a, b in zip(unet.state_dict().values(), raw.state_dict().values()))   # EMA != raw
    # the codec is frozen and is not part of the diffusion checkpoint: only the U-Net (raw + EMA) is saved
    assert set(torch.load(run / "final.pt").keys()) == {"unet", "ema", "step", "config"}


# --------------------------------------------------------------------------- decoding
class _Pointwise:
    """eps = 0.1 x + 0.2 c, applied per pixel (no spatial mixing), so tiled and whole-image decoding must agree."""
    def __call__(self, inp, tt):
        return SimpleNamespace(sample=0.1 * inp[:, :3] + 0.2 * inp[:, 3:])


def test_tile_starts_cover_the_range_and_end_exactly_at_the_border():
    for length, tile, ov in [(128, 64, 16), (200, 64, 32), (64, 64, 16), (40, 64, 8), (500, 128, 32)]:
        s = tile_starts(length, tile, ov)
        assert s[0] == 0 and s[-1] == max(0, length - tile)
        if length > tile:
            assert all(b - a <= tile - ov for a, b in zip(s, s[1:]))             # consecutive tiles overlap >= ov
            assert max(s) + tile == length


def test_tiled_decode_equals_full_decode_for_a_pointwise_network():
    sched = Schedule(1000, "cosine")
    coarse = torch.rand(1, 3, 100, 150)                                      # padded internally to 128 x 192
    full = diffusion_decode(_Pointwise(), sched, coarse, steps=5, w=1.5, seed=3)
    tiled = diffusion_decode(_Pointwise(), sched, coarse, steps=5, w=1.5, seed=3, tile=64, overlap=16)
    assert full.shape == coarse.shape and tiled.shape == coarse.shape
    assert full.min() >= 0 and full.max() <= 1
    assert torch.allclose(full, tiled, atol=1e-5)


def test_decode_is_reproducible_and_validates_tiling_arguments():
    sched = Schedule(1000)
    c = torch.rand(1, 3, 64, 64)
    a = diffusion_decode(_Pointwise(), sched, c, steps=4, seed=1)
    assert torch.equal(a, diffusion_decode(_Pointwise(), sched, c, steps=4, seed=1))
    assert not torch.equal(a, diffusion_decode(_Pointwise(), sched, c, steps=4, seed=2))
    big = torch.rand(1, 3, 128, 128)
    with pytest.raises(NotImplementedError):
        diffusion_decode(_Pointwise(), sched, big, steps=2, tile=64, guide=lambda x0, t: x0)
    with pytest.raises(ValueError):
        diffusion_decode(_Pointwise(), sched, big, steps=2, tile=60)
    with pytest.raises(ValueError):
        diffusion_decode(_Pointwise(), sched, big, steps=2, tile=64, overlap=64)


def test_decode_with_a_real_unet_runs_on_non_multiple_sizes():
    u = make_unet(64, SMALL).eval()
    out = diffusion_decode(u, Schedule(50, "cosine"), torch.rand(1, 3, 70, 90), steps=2, w=1.5, seed=0, tile=64, overlap=16)
    assert out.shape == (1, 3, 70, 90) and torch.isfinite(out).all()


# --------------------------------------------------------------------------- evaluation helpers
def test_improvement_and_target():
    assert improvement_pct(0.4, 0.3) == pytest.approx(25.0) and LPIPS_TARGET_PCT == 25.0


def test_choose_default_w_stays_in_the_recommended_range():
    sweep = pd.DataFrame(dict(w=[0.0, 1.0, 1.5, 2.0, 3.0], lpips=[0.30, 0.26, 0.24, 0.25, 0.20], psnr=[24, 23.5, 23.0, 22.5, 21]))
    w, reason = choose_default_w(sweep)
    assert w == 1.5 and "0.2400" in reason and "PSNR" in reason              # w = 3 has the best LPIPS but is outside [1, 2]
    tie = sweep.assign(lpips=[0.3, 0.25, 0.25, 0.25, 0.2])
    assert choose_default_w(tie)[0] == 1.0                                   # ties go to the smaller w
    with pytest.raises(ValueError):
        choose_default_w(sweep[sweep.w > 2.5])


def test_summary_and_checks():
    df = pd.DataFrame([dict(method=m, bpp=0.05, psnr=p, msssim=0.8, lpips=l, time=0.0)
                       for m, p, l in [("coarse", 24, 0.40), ("coarse", 25, 0.40), ("diffusion", 23, 0.28), ("diffusion", 23, 0.30),
                                       ("jpeg", 20, 0.6), ("webp", 21, 0.55)]])
    s = summarize(df)
    assert list(s.index) == ["coarse", "diffusion", "jpeg", "webp"] and s.loc["coarse", "psnr"] == 24.5
    c = checks(s, {"diffusion": 80.0, "jpeg": 120.0, "webp": 110.0, "coarse": 150.0})
    assert c["lpips_reduction_pct"] == pytest.approx(27.5) and c["lpips_target_met"] and c["fid_beats_jpeg_webp_coarse"]
    assert not checks(s, {"diffusion": 130.0, "jpeg": 120.0, "coarse": 150.0})["fid_beats_jpeg_webp_coarse"]
    assert "| method |" in fmt_table(s) and "0.2900" in fmt_table(s)


# --------------------------------------------------------------------------- snapshots and the checkpoint comparison
def test_snapshots_are_kept_beside_last_and_survive_a_resume_with_a_new_total(tmp_path):
    lst = _images(tmp_path)
    codec_path, _ = _codec_ckpt(tmp_path)
    run = tmp_path / "run"
    kw = dict(codec_path=codec_path, out=run, batch=2, crop=64, channels=SMALL, T=100, warmup=2, ckpt_every=2,
              log_every=2, workers=0, train_list=lst, device="cpu")
    train_diffusion(iters=8, stop_at=4, snapshots=(2, 4, 8), **kw)             # "30k" of a longer run
    assert (run / "snapshot_2.pt").exists() and (run / "snapshot_4.pt").exists() and not (run / "snapshot_8.pt").exists()
    train_diffusion(iters=6, snapshots=(2, 4, 6), **kw)                         # resume and shorten the total: cosine now ends at 6
    assert (run / "snapshot_6.pt").exists() and (run / "final.pt").exists()
    s = torch.load(run / "snapshot_4.pt")
    assert set(s) == {"unet", "ema", "step", "config"} and s["step"] == 4
    assert load_unet(run / "snapshot_4.pt")[1]["T"] == 100


def test_compare_checkpoints_reports_codec_only_and_every_checkpoint(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cmp_ck", Path(__file__).resolve().parents[1] / "scripts" / "compare_checkpoints.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from truesight.codec.train_codec import load_codec
    lst = _images(tmp_path, n=3, size=64)
    codec_path, _ = _codec_ckpt(tmp_path)
    run = tmp_path / "run"
    train_diffusion(codec_path=codec_path, out=run, iters=4, batch=2, crop=64, channels=SMALL, T=50, warmup=1, ckpt_every=2,
                    log_every=2, workers=0, train_list=lst, device="cpu", snapshots=(2,))
    codec, _ = load_codec(codec_path, torch.device("cpu"))
    files = open(lst).read().split()
    per, summ = mod.compare(codec, [run / "snapshot_2.pt", run / "final.pt"], files, steps=2, images=3, crop=64)
    assert summ.checkpoint.tolist() == ["codec only (no diffusion)", "step 2", "step 4"]
    assert {"lpips_change_vs_first", "ci95_low", "ci95_high"} <= set(summ.columns) and len(per) == 6
    assert summ.loc[2, "ci95_low"] <= summ.loc[2, "lpips_change_vs_first"] <= summ.loc[2, "ci95_high"]


def test_ablation_runner_evaluates_the_half_budget_checkpoint_too(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    lst = _images(tmp_path, n=3, size=64)
    codec_path, _ = _codec_ckpt(tmp_path)
    out = tmp_path / "abl"
    r = subprocess.run([sys.executable, str(root / "scripts" / "run_train_ablations.py"), "--codec", codec_path, "--out", str(out),
                        "--runs", "base", "adam", "--iters", "4", "--snapshots", "2", "4", "--batch", "2", "--crop", "64",
                        "--channels", *map(str, SMALL), "--workers", "0", "--ckpt-every", "2", "--train-list", lst,
                        "--test-list", lst, "--eval-images", "2", "--eval-crop", "64", "--steps", "2", "--device", "cpu"],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr[-1500:]
    by = pd.read_csv(out / "summary_by_step.csv")
    assert sorted(by.step.unique()) == [2, 4] and set(by.run) == {"base", "adam"}      # 4 = final (snapshot 4 is the final.pt step)
    assert (out / "base" / "snapshot_2.pt").exists() and (out / "summary_2.csv").exists() and (out / "summary.csv").exists()


def test_ablation_runner_does_not_evaluate_a_finished_run_twice(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    lst = _images(tmp_path, n=3, size=64)
    codec_path, _ = _codec_ckpt(tmp_path)
    out = tmp_path / "abl"
    cmd = [sys.executable, str(root / "scripts" / "run_train_ablations.py"), "--codec", codec_path, "--out", str(out),
           "--runs", "base", "--iters", "4", "--snapshots", "2", "--batch", "2", "--crop", "64",
           "--channels", *map(str, SMALL), "--workers", "0", "--ckpt-every", "2", "--train-list", lst,
           "--test-list", lst, "--eval-images", "2", "--eval-crop", "64", "--steps", "2", "--device", "cpu"]
    r1 = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert r1.returncode == 0, r1.stderr[-1500:]
    assert "earlier session" not in r1.stdout
    stamp = (out / "base" / "eval.csv").stat().st_mtime_ns
    r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=root)                       # the restarted session
    assert r2.returncode == 0, r2.stderr[-1500:]
    assert r2.stdout.count("earlier session") == 2                                           # final and the snapshot: both reused
    assert (out / "base" / "eval.csv").stat().st_mtime_ns == stamp                           # nothing decoded, nothing rewritten
    assert (out / "summary_by_step.csv").exists()
    r3 = subprocess.run(cmd + ["--eval-images", "1"], capture_output=True, text=True, cwd=root)   # another image set: decoded again
    assert r3.returncode == 0 and "earlier session" not in r3.stdout
    r4 = subprocess.run(cmd + ["--reeval"], capture_output=True, text=True, cwd=root)
    assert r4.returncode == 0 and "earlier session" not in r4.stdout


def test_resume_without_any_snapshot_keeps_the_resumed_state_as_a_snapshot(tmp_path):
    lst = _images(tmp_path)
    codec_path, _ = _codec_ckpt(tmp_path)
    run = tmp_path / "run"
    kw = dict(codec_path=codec_path, out=run, batch=2, crop=64, channels=SMALL, T=100, warmup=2, ckpt_every=2,
              log_every=2, workers=0, train_list=lst, device="cpu")
    train_diffusion(iters=8, stop_at=3, **kw)                                  # an old run, started before snapshots existed
    assert not list(run.glob("snapshot_*.pt"))
    train_diffusion(iters=6, snapshots=(6,), **kw)                             # resumed with a shorter total
    assert torch.load(run / "snapshot_3.pt")["step"] == 3 and (run / "snapshot_6.pt").exists()
