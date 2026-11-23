"""Time budget, crash safety, resume and storage sync (truesight/runtime.py, truesight/sync.py), plus the Step 9 warm-up,
the ReLU ablation (A2) and the A7 / A8 training options that use them."""
import time

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import truesight.codec.train_codec as tc
from truesight import runtime
from truesight.codec.eval_codec import ablation_act
from truesight.codec.train_codec import load_codec, train_codec
from truesight.codec.vae_codec import TrueSightCodec
from truesight.diffusion.schedule import Schedule
from truesight.diffusion.train_diffusion import load_unet, train_diffusion
from truesight.diffusion.uncond import (load_uncond, sample_checks, sample_grid_figure, sample_uncond, train_uncond,
                                        forward_process_figure)
from truesight.sync import Syncer, local_files, parse_spec, wanted

N, M = 16, 24


@pytest.fixture(autouse=True)
def fresh_guard():
    """Every test starts with no time limit and no pending stop signal."""
    runtime.set_guard(runtime.RunGuard(time_limit_h=None, install_signals=False))
    yield
    runtime.set_guard(runtime.RunGuard(time_limit_h=None, install_signals=False))


def _images(tmp_path, n=4, size=96):
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


KW = dict(K=3, N=N, M=M, batch=2, crop=64, workers=0, log_every=1, ckpt_every=1000, device="cpu")


# ---------------------------------------------------------------------------------------------------- the guard itself
def test_guard_budget_and_cli(monkeypatch, capsys):
    for k in ("TS_TIME_LIMIT_HOURS", "TS_SESSION_START", "TS_SAVE_EVERY_MIN"):      # the pipeline that runs the tests sets these
        monkeypatch.delenv(k, raising=False)
    g = runtime.RunGuard(time_limit_h=1.0, start=time.time() - 3600 * 0.5, install_signals=False)
    assert 25 < g.remaining_min() < 35 and not g.stop_requested()
    assert runtime.RunGuard(time_limit_h=1.0, start=time.time() - 3700, install_signals=False).stop_requested()
    assert runtime.RunGuard(time_limit_h=None, install_signals=False).remaining_min() == float("inf")
    monkeypatch.setenv("TS_SESSION_START", str(time.time() - 3600 * 11))
    monkeypatch.setenv("TS_TIME_LIMIT_HOURS", "11.5")
    assert runtime._cli(["need", "10"]) == 0                         # ~30 min left
    assert runtime._cli(["need", "45"]) == runtime.EXIT_PAUSED
    assert runtime._cli(["remaining"]) == 0 and 25 < float(capsys.readouterr().out.split()[-1]) < 31


def test_run_cli_turns_pause_into_exit_75():
    def main():
        raise runtime.Paused("budget used up")
    with pytest.raises(SystemExit) as e:
        runtime.run_cli(main)
    assert e.value.code == runtime.EXIT_PAUSED


def test_save_due_is_time_based():
    g = runtime.RunGuard(save_every_min=0.0001, install_signals=False)
    time.sleep(0.02)
    assert g.save_due()
    g.saved()
    g2 = runtime.RunGuard(save_every_min=60, install_signals=False)
    assert not g2.save_due()


# ---------------------------------------------------------------------------------- budget pause then continue (codec)
def test_time_budget_pauses_then_continues_exactly(tmp_path):
    lst = _images(tmp_path)
    run = tmp_path / "run"
    runtime.set_guard(runtime.RunGuard(time_limit_h=0, install_signals=False))       # budget already used up
    with pytest.raises(runtime.Paused):
        train_codec(0.013, run, iters=6, train_list=lst, **KW)
    s = torch.load(run / "last.pt")
    assert s["it"] == 1 and not (run / "final.pt").exists()                          # saved after the first step
    runtime.set_guard(runtime.RunGuard(time_limit_h=None, install_signals=False))    # "new session"
    train_codec(0.013, run, iters=6, train_list=lst, **KW)
    assert (run / "final.pt").exists()
    assert pd.read_csv(run / "log.csv")["it"].tolist() == [1, 2, 3, 4, 5, 6]          # continued from 1, never from 0


# -------------------------------------------------------------------------- a bug mid-run: state saved, then continue
def test_crash_saves_exact_state_and_run_continues(tmp_path, monkeypatch):
    lst = _images(tmp_path)
    run = tmp_path / "run"
    real = tc.rd_loss
    calls = {"n": 0}

    def buggy(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("simulated bug in the loop")
        return real(*a, **k)

    monkeypatch.setattr(tc, "rd_loss", buggy)
    with pytest.raises(RuntimeError, match="simulated bug"):
        train_codec(0.013, run, iters=8, train_list=lst, **KW)                       # ckpt_every=1000: only the crash saves
    assert torch.load(run / "last.pt")["it"] == 4                                     # exactly where it stopped
    monkeypatch.setattr(tc, "rd_loss", real)                                          # "fixed the bug"
    train_codec(0.013, run, iters=8, train_list=lst, **KW)
    assert (run / "final.pt").exists()
    assert pd.read_csv(run / "log.csv")["it"].tolist() == list(range(1, 9))


def test_crash_with_non_finite_weights_keeps_previous_checkpoint(tmp_path, monkeypatch):
    lst = _images(tmp_path)
    run = tmp_path / "run"
    kw = {**KW, "ckpt_every": 2}
    real = tc.rd_loss
    calls = {"n": 0}

    def nan_then_raise(out, x, lmbda):
        calls["n"] += 1
        if calls["n"] == 3:
            for p in nets[0].parameters():
                p.data.fill_(float("nan")); break
            raise FloatingPointError("weights went NaN")
        return real(out, x, lmbda)

    nets = []
    orig_init = TrueSightCodec.__init__

    def spy(self, *a, **k):
        orig_init(self, *a, **k)
        nets.append(self)
    monkeypatch.setattr(TrueSightCodec, "__init__", spy)
    monkeypatch.setattr(tc, "rd_loss", nan_then_raise)
    with pytest.raises(FloatingPointError):
        train_codec(0.013, run, iters=8, train_list=lst, **kw)
    s = torch.load(run / "last.pt")
    assert s["it"] == 2                                                               # the periodic save, not the NaN state
    assert all(torch.isfinite(v).all() for v in s["net"].values() if v.dtype.is_floating_point)


# --------------------------------------------------------------------- A2 ReLU codec, A7 / A8 diffusion options
def test_relu_codec_trains_saves_and_reloads(tmp_path):
    lst = _images(tmp_path)
    run = tmp_path / "relu"
    train_codec(0.013, run, iters=2, train_list=lst, act="relu", **KW)
    net, cfg = load_codec(run / "final.pt")
    assert cfg["act"] == "relu" and net.act == "relu"
    assert not any(type(m).__name__ == "GDN" for m in net.g_a) and any(isinstance(m, torch.nn.ReLU) for m in net.g_a)
    out = net.eval()(torch.rand(1, 3, 64, 64))
    assert out["x_hat"].shape == (1, 3, 64, 64)
    with pytest.raises(ValueError):
        TrueSightCodec(N=N, M=M, act="tanh")
    with pytest.raises(ValueError):                                                   # a ReLU run cannot start from a GDN base
        train_codec(0.0035, tmp_path / "x", iters=1, train_list=lst, init_from=str(run / "final.pt"), act="gdn", **KW)


def test_gdn_checkpoint_without_act_field_still_loads(tmp_path):
    lst = _images(tmp_path)
    run = tmp_path / "g"
    train_codec(0.013, run, iters=1, train_list=lst, **KW)
    s = torch.load(run / "final.pt")
    del s["config"]["act"]                                                            # a checkpoint from before A2 existed
    torch.save(s, run / "old.pt")
    assert load_codec(run / "old.pt")[0].act == "gdn"


def test_ablation_act_table():
    gdn = pd.DataFrame(dict(lmbda=[0.0018, 0.0130, 0.0483], bpp=[0.1, 0.4, 1.2], psnr=[26.0, 31.0, 36.0]))
    relu = pd.DataFrame(dict(lmbda=[0.0130], bpp=[0.5], psnr=[30.0]))
    r = ablation_act(gdn, relu).iloc[0]
    assert r.d_psnr_same_lambda == pytest.approx(1.0) and r.d_bpp_pct_same_lambda == pytest.approx(-20.0)
    assert r.saving_pct_equal_psnr < 0                                                # GDN needs fewer bits for PSNR 30


def _tiny_codec(tmp_path, lst):
    run = tmp_path / "codec"
    train_codec(0.0018, run, iters=1, train_list=lst, **KW)
    return str(run / "final.pt")


def test_diffusion_ablation_options_and_crash_resume(tmp_path, monkeypatch):
    lst = _images(tmp_path)
    codec = _tiny_codec(tmp_path, lst)
    kw = dict(batch=2, crop=64, channels=(16, 32, 32, 32), workers=0, train_list=lst, log_every=1, ckpt_every=1000,
              device="cpu", amp=False, warmup=1)
    train_diffusion(codec, tmp_path / "adam", iters=3, optimizer="adam", **kw)
    train_diffusion(codec, tmp_path / "const", iters=3, lr_schedule="constant", **kw)
    train_diffusion(codec, tmp_path / "lin", iters=3, kind="linear", **kw)
    _, c1 = load_unet(tmp_path / "adam" / "final.pt")
    _, c2 = load_unet(tmp_path / "const" / "final.pt")
    _, c3 = load_unet(tmp_path / "lin" / "final.pt")
    assert (c1["optimizer"], c2["lr_schedule"], c3["kind"]) == ("adam", "constant", "linear")
    lrs = pd.read_csv(tmp_path / "const" / "log.csv")["lr"]
    assert lrs.nunique() == 1 and lrs.iloc[0] == pytest.approx(1e-4)                  # constant, no warm-up or decay
    with pytest.raises(ValueError):
        train_diffusion(codec, tmp_path / "bad", iters=1, optimizer="sgd", **kw)

    import truesight.diffusion.train_diffusion as td
    real, n = td.drop_condition, {"n": 0}

    def boom(c, p, generator=None):
        n["n"] += 1
        if n["n"] == 3:
            raise RuntimeError("bug at step 3")
        return real(c, p, generator)
    monkeypatch.setattr(td, "drop_condition", boom)
    with pytest.raises(RuntimeError):
        train_diffusion(codec, tmp_path / "crash", iters=6, **kw)
    assert torch.load(tmp_path / "crash" / "last.pt")["step"] == 2
    monkeypatch.setattr(td, "drop_condition", real)
    train_diffusion(codec, tmp_path / "crash", iters=6, **kw)
    assert pd.read_csv(tmp_path / "crash" / "log.csv")["step"].tolist() == [1, 2, 3, 4, 5, 6]


# ------------------------------------------------------------------------------------------------ Step 9 warm-up DDPM
def test_uncond_ddpm_trains_resumes_and_draws_figures(tmp_path):
    lst = _images(tmp_path)
    out = tmp_path / "u"
    kw = dict(batch=2, crop=64, channels=(16, 32, 32, 32), workers=0, train_list=lst, log_every=2, ckpt_every=1000,
              device="cpu", amp=False, warmup=1)
    runtime.set_guard(runtime.RunGuard(time_limit_h=0, install_signals=False))
    with pytest.raises(runtime.Paused):
        train_uncond(out, iters=4, **kw)
    runtime.set_guard(runtime.RunGuard(time_limit_h=None, install_signals=False))
    train_uncond(out, iters=4, **kw)
    assert (out / "final.pt").exists()
    unet, cfg = load_uncond(out / "final.pt")
    sched = Schedule(cfg["T"], cfg["kind"])
    s = sample_uncond(unet, sched, n=4, size=64, steps=3)
    assert s.shape == (4, 3, 64, 64) and torch.isfinite(s).all() and s.abs().max() <= 1.0 + 1e-5
    x0 = torch.rand(1, 3, 64, 64) * 2 - 1
    forward_process_figure(x0, sched, tmp_path / "fwd.png")
    sample_grid_figure(s, tmp_path / "grid.png")
    assert (tmp_path / "fwd.png").stat().st_size > 1000 and (tmp_path / "grid.png").stat().st_size > 1000


def test_sample_checks_tell_noise_flat_and_structure_apart():
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(8, 3, 32, 32, generator=g).clamp(-1, 1)
    flat = torch.zeros(8, 3, 32, 32)
    xs = torch.linspace(-1, 1, 32)
    struct = (xs[None, :] + xs[:, None]).div(2)[None, None].repeat(8, 3, 1, 1)
    assert not sample_checks(noise)["not_pure_noise"] and sample_checks(noise)["not_flat"]
    assert not sample_checks(flat)["not_flat"]
    ok = sample_checks(struct)
    assert ok["not_flat"] and ok["not_pure_noise"]


# ---------------------------------------------------------------------------------------------------------- storage sync
def test_wanted_filter():
    assert wanted("step7_codec/lmbda0.0130/last.pt") and wanted(".done/codec") and wanted("a/log.csv")
    assert not wanted("a/last.pt.tmp") and not wanted("a/__pycache__/x.pt") and not wanted("a/image.jpg")
    assert not wanted("a/huge.pt", size=10 * 2 ** 30)


def test_parse_spec():
    assert parse_spec("hf:user/repo") == ("hf", "user/repo") and parse_spec("dir:/tmp/x") == ("dir", "/tmp/x")
    for bad in ("user/repo", "ftp:x", "hf:"):
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_dir_sync_round_trip_resumes_on_a_fresh_machine(tmp_path):
    run, store, fresh = tmp_path / "experiments", tmp_path / "persist", tmp_path / "new_experiments"
    (run / "step7" / "lmbda0.0130").mkdir(parents=True)
    (run / ".done").mkdir()
    (run / "step7" / "lmbda0.0130" / "last.pt").write_bytes(b"checkpoint-v1")
    (run / "step7" / "lmbda0.0130" / "last.pt.tmp").write_bytes(b"half written")
    (run / "step7" / "lmbda0.0130" / "log.csv").write_text("it,loss\n1,2\n")
    (run / ".done" / "data").write_text("")
    s = Syncer(f"dir:{store}", run)
    assert s.push() == 3
    assert not list(store.rglob("*.tmp"))
    assert s.push() == 0                                                              # nothing changed: nothing copied
    time.sleep(1.1)
    (run / "step7" / "lmbda0.0130" / "last.pt").write_bytes(b"checkpoint-v2-longer")
    assert s.push() == 1
    new = Syncer(f"dir:{store}", fresh)                                               # the next Kaggle session
    assert new.pull() == 3
    assert (fresh / "step7" / "lmbda0.0130" / "last.pt").read_bytes() == b"checkpoint-v2-longer"
    assert (fresh / ".done" / "data").exists() and set(local_files(fresh)) == set(local_files(run))


def test_background_sync_pushes_and_survives_failures(tmp_path):
    run = tmp_path / "e"
    run.mkdir()
    (run / "a.csv").write_text("x\n")
    s = Syncer(f"dir:{tmp_path / 'p'}", run).start_background(interval_min=0.0005)
    time.sleep(0.5)
    s.stop_background()
    assert (tmp_path / "p" / "a.csv").exists()
