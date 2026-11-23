"""The Kaggle driver: pause at the time limit, continue in a fresh session, survive a crash (truesight/kaggle.py).

These tests run the real trainer command line in subprocesses, with the run state stored through a ``dir:`` backend that stands
in for Hugging Face, and a new empty experiments folder for every "session" (a new Kaggle machine has nothing local)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from truesight import kaggle, runtime

ROOT = Path(__file__).resolve().parents[1]


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


def _train_cmd(lst, out, iters=40):
    return [sys.executable, "-m", "truesight.codec.train_codec", "--lmbda", "0.013", "--out", str(out), "--iters", str(iters), "--N", "16",
            "--M", "24", "--batch", "2", "--crop", "64", "--workers", "0", "--ckpt-every", "1000", "--train-list", lst]


def _session(tmp_path, name, store, cmd, hours):
    exp = tmp_path / name
    exp.mkdir(exist_ok=True)
    return exp, kaggle.run_session(ROOT, f"dir:{store}", hours=hours, cmd=cmd, exp_dir=exp, do_git_pull=False, install=False)


def test_classify():
    assert kaggle.classify(0) == kaggle.DONE and kaggle.classify(75) == kaggle.PAUSED
    for rc in (1, 2, 137, -9):
        assert kaggle.classify(rc) == kaggle.CRASHED


def test_session_env():
    env = kaggle.session_env("hf:u/r", 11.5, start=123.0, data_repo="u/d", base={"A": "1"})
    assert env["TS_TIME_LIMIT_HOURS"] == "11.5" and env["TS_SESSION_START"] == "123.0" and env["TS_SYNC"] == "hf:u/r"
    assert env["TS_DATA_REPO"] == "u/d" and env["A"] == "1"


def test_missing_packages_only_reports_what_is_absent():
    req = {"os": "os", "surely_not_installed_module_xyz": "surely-not-installed"}
    assert kaggle.missing_packages(req) == ["surely-not-installed"]
    assert kaggle.ensure_requirements({"os": "os"}) == []


def test_prepare_data_links_an_attached_dataset(tmp_path):
    inp = tmp_path / "input" / "sphota-data"
    (inp / "raw" / "kodak" / "images").mkdir(parents=True)
    (inp / "raw" / "kodak" / "images" / "a.png").write_bytes(b"x")
    (inp / "processed").mkdir()
    (inp / "processed" / "train.txt").write_text("data/raw/kodak/images/a.png\n")
    root = tmp_path / "proj"
    root.mkdir()
    try:
        msg = kaggle.prepare_data(root, data_input=inp)
    except OSError:
        pytest.skip("symlinks are not allowed on this machine")
    assert "data/raw ->" in msg and (root / "data" / "raw" / "kodak" / "images" / "a.png").exists()
    assert (root / "data" / "processed" / "train.txt").read_text().startswith("data/raw/kodak")
    assert "no data source" in kaggle.prepare_data(tmp_path / "other")


def test_pause_at_time_limit_then_continue_in_a_fresh_session(tmp_path):
    lst, store = _images(tmp_path), tmp_path / "persistent_storage"
    exp1, (status1, rc1) = _session(tmp_path, "session1", store, _train_cmd(lst, tmp_path / "session1" / "run"), hours=0.0005)
    assert (status1, rc1) == (kaggle.PAUSED, runtime.EXIT_PAUSED)                          # the budget ran out mid-training
    saved = torch.load(store / "run" / "last.pt")
    assert 1 <= saved["it"] < 40 and not (store / "run" / "final.pt").exists()
    assert not (tmp_path / "session2").exists()                                           # session 2 starts on an empty machine

    exp2, (status2, rc2) = _session(tmp_path, "session2", store, _train_cmd(lst, tmp_path / "session2" / "run"), hours=5)
    assert (status2, rc2) == (kaggle.DONE, 0)
    assert (store / "run" / "final.pt").exists()
    # the second session printed "resumed from iteration N" and finished the remaining steps: the final checkpoint says 40
    assert torch.load(store / "run" / "final.pt")["it"] == 40


CRASHING_TRAINER = """
import sys, torch
sys.path.insert(0, {root!r})
import truesight.codec.train_codec as tc
real, n = tc.rd_loss, {{"n": 0}}
def buggy(*a, **k):
    n["n"] += 1
    if n["n"] == 6:
        raise KeyError("bug in the training loop at step 6")
    return real(*a, **k)
tc.rd_loss = buggy
tc.train_codec(0.013, {out!r}, iters=40, N=16, M=24, batch=2, crop=64, workers=0, ckpt_every=1000, train_list={lst!r}, device="cpu")
"""


def test_crash_keeps_exact_state_and_the_fix_continues_from_there(tmp_path):
    lst, store = _images(tmp_path), tmp_path / "persistent_storage"
    out1 = tmp_path / "crash1" / "run"
    cmd = [sys.executable, "-c", CRASHING_TRAINER.format(root=str(ROOT), out=str(out1), lst=lst)]
    _, (status, rc) = _session(tmp_path, "crash1", store, cmd, hours=5)
    assert status == kaggle.CRASHED and rc != 0
    saved = torch.load(store / "run" / "last.pt")
    assert saved["it"] == 5                                                               # exactly the 5 completed steps, saved at the crash
    assert not (store / "run" / "final.pt").exists()

    # "I fixed the bug": a new session on a new machine runs the real trainer
    _, (status2, rc2) = _session(tmp_path, "crash2", store, _train_cmd(lst, tmp_path / "crash2" / "run"), hours=5)
    assert (status2, rc2) == (kaggle.DONE, 0)
    assert (store / "run" / "final.pt").exists()
    assert torch.load(store / "run" / "final.pt")["it"] == 40


def test_pipeline_script_pauses_with_status_75_and_does_not_mark_the_stage_done(tmp_path):
    """The shell pipeline itself: with no time left a stage that needs time pauses with 75 and leaves no done marker."""
    import os
    import shutil
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    env = dict(os.environ, STAGES="baselines", SMOKE="1", TS_TIME_LIMIT_HOURS="0.0001", TS_SESSION_START="1")
    r = subprocess.run(["bash", "scripts/run_full_pipeline.sh"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    try:
        assert r.returncode == runtime.EXIT_PAUSED, r.stdout[-1500:] + r.stderr[-1500:]
        assert not (ROOT / "experiments_smoke" / ".done" / "baselines").exists()
    finally:
        shutil.rmtree(ROOT / "experiments_smoke", ignore_errors=True)
        shutil.rmtree(ROOT / "report_smoke", ignore_errors=True)
