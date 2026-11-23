"""Automatic restarts: notebook files, the next-step decision, the quota-reset arithmetic, the waiter and every safeguard
(truesight/chain.py, truesight/kaggle.py, scripts/kaggle_chain.py). The kaggle CLI is replaced by a fake that records the pushes."""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from truesight import chain, kaggle, sync

ROOT = Path(__file__).resolve().parents[1]
BASE = ["--sync", "hf:u/r", "--data-repo", "u/d", "--stages", "data", "--weekly-quota", "30", "--quota-reset-dow", "0", "--chain"]


def utc(y, mo, d, h=0, mi=0):
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp()


class FakePush:
    """Records the notebooks that would be pushed (reads them back from the folder, like Kaggle would)."""

    def __init__(self, ok=True):
        self.ok, self.pushed = ok, []

    def __call__(self, folder):
        meta = json.loads((Path(folder) / "kernel-metadata.json").read_text())
        nb = json.loads((Path(folder) / "run.ipynb").read_text())
        code = "".join(nb["cells"][0]["source"])
        args = json.loads(eval(code.split("json.loads(", 1)[1].split(")\n", 1)[0]))   # the literal passed to json.loads in the cell
        self.pushed.append((meta, args))
        return self.ok, "Kernel version 1 successfully pushed" if self.ok else "error: denied"


def cfg(**kw):
    d = dict(owner="me", kind="gpu", count=0, weekly_quota=30.0, reset=(0, 0), max_chain=40, hours_limit=11.5)
    d.update(kw)
    return chain.ChainConfig(**d)


def flag(args, name):
    return args[args.index(name) + 1]


# ------------------------------------------------------------------------------------------------ files and arguments
def test_notebook_files_have_the_right_fields_and_no_secret(tmp_path):
    for gpu in (True, False):
        folder = chain.write_kernel(tmp_path / f"k{gpu}", "me", "sphota-train", ["--sync", "hf:u/r"], gpu=gpu)
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta["id"] == "me/sphota-train" and meta["title"] == "sphota train" and meta["code_file"] == "run.ipynb"
        assert meta["kernel_type"] == "notebook" and meta["enable_internet"] == "true" and meta["is_private"] == "true"
        assert meta["enable_gpu"] == ("true" if gpu else "false") and (meta.get("machine_shape") == "NvidiaTeslaT4") == gpu
        nb = json.loads((folder / "run.ipynb").read_text())
        assert nb["nbformat"] == 4 and nb["cells"][0]["cell_type"] == "code"
        src = "".join(nb["cells"][0]["source"])
        compile(src, "run.ipynb", "exec")                                       # the generated cell is valid Python
        assert "kaggle_run.py" in src and "hf:u/r" in src
        assert "hf_" not in src.replace("hf:u/r", "") and "KEY" not in src and "get_secret('GITHUB_TOKEN')" in src   # only reads a secret
    assert "machine_shape" not in chain.kernel_metadata("me", "x", True, machine_shape="")


def test_strip_flag_and_next_argv():
    a = ["--sync", "x", "--quota-hours", "5", "--chain-count=3", "--stages", "data", "--chain-kind", "cpu_data", "--smoke"]
    assert chain.strip_flag(a, "--quota-hours") == ["--sync", "x", "--chain-count=3", "--stages", "data", "--chain-kind", "cpu_data", "--smoke"]
    assert chain.strip_flag(a, "--chain-count") == [x for x in a if x != "--chain-count=3"]
    assert chain.strip_flag(a, "--smoke", has_value=False)[-1] == "cpu_data"
    n = chain.next_argv(a, "gpu", 4, quota_hours=7.25, drop_stages=True)
    assert "--stages" not in n and "data" not in n and flag(n, "--quota-hours") == "7.250" and flag(n, "--chain-kind") == "gpu"
    assert flag(n, "--chain-count") == "4" and n.count("--chain-kind") == 1 and "--smoke" in n and flag(n, "--sync") == "x"
    w = chain.next_argv(a, "wait", 1, wait_until=1234567890.4)
    assert flag(w, "--wait-until") == "1234567890" and "--quota-hours" not in w                # stale explicit quota is dropped


# ------------------------------------------------------------------------------------------------ quota reset arithmetic
def test_last_and_next_reset():
    sat = utc(2026, 11, 21, 12)                                                  # Saturday 21 Nov 2026
    assert kaggle.last_reset(sat, 0, 0) == utc(2026, 11, 16)                      # Monday 00:00 UTC before it
    assert kaggle.next_reset(sat, 0, 0) == utc(2026, 11, 23)
    mon_early = utc(2026, 11, 23, 0, 30)
    assert kaggle.last_reset(mon_early, 0, 6) == utc(2026, 11, 16, 6)             # today's 06:00 reset has not happened yet
    assert kaggle.last_reset(utc(2026, 11, 23, 7), 0, 6) == utc(2026, 11, 23, 6)
    assert kaggle.last_reset(sat, 5, 0) == utc(2026, 11, 21)                      # a Saturday reset, same day


def test_ledger_counts_only_since_the_last_reset():
    now = utc(2026, 11, 24, 3)                                                    # Tuesday 03:00, reset Monday 00:00
    e = [{"start": utc(2026, 11, 20, 1), "reserved_h": 11.5, "hours_used": 11.0},  # last week: free again after the reset
         {"start": utc(2026, 11, 23, 5), "reserved_h": 11.5, "hours_used": 11.0}]  # this week: counts
    assert kaggle.used_quota_hours(e, now, since=kaggle.last_reset(now, 0, 0)) == pytest.approx(11.0)
    assert kaggle.used_quota_hours(e, now) == pytest.approx(22.0)                 # the rolling 7 days would still count both
    b, why = kaggle.plan_budget(11.5, weekly_quota=20, entries=e, now=now, reset=(0, 0))
    assert b == pytest.approx(20 - 11.0 - 0.5) and "since the last weekly reset" in why
    assert kaggle.plan_budget(11.5, weekly_quota=20, entries=e, now=now)[0] == 0.0           # rolling 7 days: 22 h counted, nothing left


# ------------------------------------------------------------------------------------------------ the decision
@pytest.mark.parametrize("kind,status,can_wait,expect", [
    ("cpu_data", kaggle.DONE, False, "gpu"), ("cpu_data", kaggle.PAUSED, False, "cpu_data"), ("cpu_data", kaggle.CRASHED, True, "stop"),
    ("gpu", kaggle.PAUSED, False, "gpu"), ("gpu", kaggle.DONE, True, "stop"), ("gpu", kaggle.CRASHED, True, "stop"),
    ("gpu", kaggle.NO_QUOTA, True, "wait"), ("gpu", kaggle.NO_QUOTA, False, "stop"),
])
def test_decide(kind, status, can_wait, expect):
    assert chain.decide(kind, status, 0, 40, can_wait)[0] == expect


def test_decide_safeguards():
    assert chain.decide("gpu", kaggle.PAUSED, 39, 40, True)[0] == "gpu" and chain.decide("gpu", kaggle.PAUSED, 40, 40, True)[0] == "stop"
    assert chain.decide("gpu", kaggle.PAUSED, 0, 40, True, progress_h=0.01)[0] == "stop"          # a pause with no progress never loops
    assert chain.decide("gpu", kaggle.PAUSED, 0, 40, True, progress_h=5.0)[0] == "gpu"
    assert chain.decide("gpu", kaggle.PAUSED, 0, 40, True, stopped=True)[0] == "stop"             # kill switch


# ------------------------------------------------------------------------------------------------ after a session
def run_after(tmp_path, config, status, argv=None, last=None, push=None, now=None, spec=None, ready=True):
    push = push or FakePush()
    exp = tmp_path / "exp"; exp.mkdir(exist_ok=True)
    action = chain.after_session(config, argv or BASE, status, exp, spec or f"dir:{tmp_path / 'store'}", tmp_path / "k", push=push,
                                 now=now or utc(2026, 11, 21, 12), last=last or {"gpu": True, "hours_used": 11.0}, log=lambda m: None,
                                 ready=ready)
    return action, push


def test_gpu_session_that_paused_starts_the_next_gpu_session(tmp_path):
    action, push = run_after(tmp_path, cfg(count=2, quota_hours=20.0), kaggle.PAUSED)
    assert action == "gpu" and len(push.pushed) == 1
    meta, args = push.pushed[0]
    assert meta["id"] == "me/sphota-train" and meta["enable_gpu"] == "true"
    assert flag(args, "--chain-count") == "3" and flag(args, "--chain-kind") == "gpu"
    assert float(flag(args, "--quota-hours")) == pytest.approx(9.0)                # 20 h minus the 11 h this session used


def test_data_notebook_hands_over_to_the_gpu_notebook_with_all_stages(tmp_path):
    action, push = run_after(tmp_path, cfg(kind="cpu_data"), kaggle.DONE, last={"gpu": False, "hours_used": 1.5})
    assert action == "gpu"
    meta, args = push.pushed[0]
    assert meta["id"] == "me/sphota-train" and meta["enable_gpu"] == "true"
    assert "--stages" not in args and flag(args, "--chain-kind") == "gpu"           # the GPU notebook runs the whole pipeline


def test_no_chaining_after_a_crash_or_when_everything_is_done(tmp_path):
    for status in (kaggle.CRASHED, kaggle.DONE):
        action, push = run_after(tmp_path, cfg(), status)
        assert action == "stop" and push.pushed == []


def test_out_of_quota_starts_a_cpu_waiter_that_wakes_after_the_reset(tmp_path):
    now = utc(2026, 11, 21, 12)
    action, push = run_after(tmp_path, cfg(), kaggle.NO_QUOTA, now=now)
    assert action == "wait"
    meta, args = push.pushed[0]
    assert meta["id"] == "me/sphota-wait" and meta["enable_gpu"] == "false"            # a CPU notebook: no GPU quota while it sleeps
    assert float(flag(args, "--wait-until")) == pytest.approx(utc(2026, 11, 23, 0, 10))   # Monday 00:00 UTC + 10 minutes
    action, push = run_after(tmp_path, cfg(reset=None), kaggle.NO_QUOTA)                 # reset unknown: cannot wait
    assert action == "stop" and push.pushed == []


def test_the_quota_is_checked_before_a_start(tmp_path):
    exp = tmp_path / "exp"; exp.mkdir()
    kaggle.write_ledger_entry(exp, "s1", utc(2026, 11, 21, 1), 11.5, hours_used=11.2)
    kaggle.write_ledger_entry(exp, "s2", utc(2026, 11, 20, 1), 11.5, hours_used=11.3)      # 22.5 h used this week
    kaggle.write_ledger_entry(exp, "s3", utc(2026, 11, 19, 1), 11.5, hours_used=7.0)        # 29.5 h in total, since Monday
    action, push = run_after(tmp_path, cfg(), kaggle.PAUSED)
    assert action == "wait" and push.pushed[0][0]["id"] == "me/sphota-wait"
    action, push = run_after(tmp_path, cfg(reset=None), kaggle.PAUSED)                    # no reset time: stop instead of starting
    assert action == "stop" and push.pushed == []


def test_kill_switch_and_no_progress_and_chain_limit(tmp_path):
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / ".stop_chain").write_text("")
    assert run_after(tmp_path, cfg(), kaggle.PAUSED)[0] == "stop"                         # .stop_chain in the run repo
    (tmp_path / "store" / ".stop_chain").unlink()
    assert run_after(tmp_path, cfg(), kaggle.PAUSED, last={"gpu": True, "hours_used": 0.02})[0] == "stop"
    assert run_after(tmp_path, cfg(count=40), kaggle.PAUSED)[0] == "stop"
    assert run_after(tmp_path, cfg(), kaggle.PAUSED)[0] == "gpu"                           # and with none of them it goes on


def test_a_failed_push_stops_the_chain_without_raising(tmp_path):
    action, push = run_after(tmp_path, cfg(), kaggle.PAUSED, push=FakePush(ok=False))
    assert action == "stop"


# ------------------------------------------------------------------------------------------------ the waiter
def test_wait_until_with_a_fake_clock():
    clock = {"t": 0.0}
    sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    now = lambda: clock["t"]
    assert chain.wait_until(500.0, max_hours=1, sleep=sleep, now=now, step_s=60) is True and clock["t"] >= 500
    clock["t"] = 0.0
    assert chain.wait_until(10 ** 6, max_hours=1, sleep=sleep, now=now, step_s=600) is False and 3600 <= clock["t"] < 4300   # the limit first


def test_waiter_starts_the_gpu_notebook_after_the_reset_or_another_waiter(tmp_path):
    clock = {"t": utc(2026, 11, 21, 12)}
    sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    now = lambda: clock["t"]
    soon = clock["t"] + 3 * 3600
    p = FakePush()
    assert chain.run_wait(cfg(kind="wait", wait_until=soon, count=3), BASE + ["--chain-kind", "wait", "--wait-until", str(soon)],
                          f"dir:{tmp_path}", tmp_path / "k", push=p, sleep=sleep, now=now, log=lambda m: None) == "gpu"
    meta, args = p.pushed[0]
    assert meta["id"] == "me/sphota-train" and "--wait-until" not in args and flag(args, "--chain-count") == "4"
    far = clock["t"] + 3 * 86400                                                          # reset is days away: wait again
    p = FakePush()
    assert chain.run_wait(cfg(kind="wait", wait_until=far), BASE, f"dir:{tmp_path}", tmp_path / "k", push=p, sleep=sleep, now=now,
                          log=lambda m: None) == "wait"
    assert p.pushed[0][0]["id"] == "me/sphota-wait" and float(flag(p.pushed[0][1], "--wait-until")) == pytest.approx(far)
    (tmp_path / ".stop_chain").write_text("")
    p = FakePush()
    assert chain.run_wait(cfg(kind="wait", wait_until=clock["t"]), BASE, f"dir:{tmp_path}", tmp_path / "k", push=p, sleep=sleep, now=now,
                          log=lambda m: None) == "stop" and p.pushed == []


# ------------------------------------------------------------------------------------------------ push command, sessions, misc
def test_push_kernel_command_and_error_handling(tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="Kernel version 3 successfully pushed", stderr="")
    ok, out = chain.push_kernel(tmp_path, run=fake_run)
    assert ok and calls[0][:3] == ["kaggle", "kernels", "push"] and calls[0][-2:] == ["-p", str(tmp_path)]
    assert chain.push_kernel(tmp_path, run=lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="401 Unauthorized"))[0] is False
    assert chain.push_kernel(tmp_path, run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="Error: bad slug", stderr=""))[0] is False

    def missing(*a, **k):
        raise FileNotFoundError("kaggle")
    assert chain.push_kernel(tmp_path, run=missing)[0] is False


def test_stop_file_is_synced_and_found(tmp_path):
    assert sync.wanted(".stop_chain")
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / ".stop_chain").write_text("")
    assert chain.stop_requested(f"dir:{tmp_path / 'x'}") and not chain.stop_requested(f"dir:{tmp_path}")


def test_cpu_session_uses_no_quota_and_writes_no_ledger(tmp_path):
    exp, store = tmp_path / "exp", tmp_path / "store"
    exp.mkdir()
    kaggle.write_ledger_entry(exp, "old", kaggle.time.time() - 3600 * 40, 30.0)             # the GPU week is used up ...
    code = "import os; print('LIMIT', os.environ['TS_TIME_LIMIT_HOURS'])"
    status, rc = kaggle.run_session(ROOT, f"dir:{store}", hours=11.5, cmd=[sys.executable, "-c", code], exp_dir=exp, do_git_pull=False,
                                    install=False, weekly_quota=30, gpu=False)
    assert (status, rc) == (kaggle.DONE, 0)                                                 # ... but a CPU session still runs
    assert len(kaggle.read_ledger(exp)) == 1 and kaggle.LAST_SESSION["gpu"] is False and kaggle.LAST_SESSION["budget_h"] == 11.5


def test_start_script_writes_a_notebook_without_pushing(tmp_path):
    r = subprocess.run([sys.executable, "scripts/kaggle_chain.py", "start", "--kind", "cpu_data", "--owner", "me", "--workdir", str(tmp_path),
                        "--write-only", "--", "--sync", "hf:u/r", "--stages", "data"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "sphota-data" / "kernel-metadata.json").read_text())
    assert meta["id"] == "me/sphota-data" and meta["enable_gpu"] == "false"
    src = "".join(json.loads((tmp_path / "sphota-data" / "run.ipynb").read_text())["cells"][0]["source"])
    assert "--chain-kind" in src and "--chain-owner" in src and "--chain" in src
    assert subprocess.run([sys.executable, "scripts/kaggle_chain.py", "stop-help"], cwd=ROOT, capture_output=True, text=True).returncode == 0


def test_driver_refuses_to_start_without_a_hugging_face_token():
    import os
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["TS_NO_KAGGLE_SECRETS"] = "1"          # on Kaggle the notebook's own secrets would otherwise be loaded and the session would really start
    r = subprocess.run([sys.executable, "scripts/kaggle_run.py", "--sync", "hf:u/r", "--no-git-pull", "--no-install"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "HF_TOKEN not found" in (r.stderr + r.stdout)


def test_kaggle_credentials_detection(tmp_path):
    assert not chain.have_kaggle_credentials({}, home=tmp_path)
    assert chain.have_kaggle_credentials({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, home=tmp_path)
    assert not chain.have_kaggle_credentials({"KAGGLE_USERNAME": "u"}, home=tmp_path)
    assert chain.have_kaggle_credentials({"KAGGLE_API_TOKEN": "t"}, home=tmp_path)
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    assert chain.have_kaggle_credentials({}, home=tmp_path)


def test_cli_is_upgraded_once_only_for_the_new_token_type():
    calls = []
    fake = lambda cmd, **kw: calls.append(cmd)
    chain._CLI_READY["done"] = False
    assert chain.ensure_cli({}, run=fake) is False and calls == []                       # no KAGGLE_API_TOKEN: nothing to do
    assert chain.ensure_cli({"KAGGLE_API_TOKEN": "KGAT_x"}, run=fake) is True
    assert calls[0][1:4] == ["-m", "pip", "install"] and "kaggle" in calls[0]
    assert chain.ensure_cli({"KAGGLE_API_TOKEN": "KGAT_x"}, run=fake) is False and len(calls) == 1   # only once
    chain._CLI_READY["done"] = False


def _fresh_clone(root):
    """What a git clone of the repository leaves in data/: tracked placeholder and split-list files."""
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / ".gitkeep").write_text("")
    (root / "data" / "processed").mkdir()
    (root / "data" / "processed" / ".gitkeep").write_text("")
    (root / "data" / "processed" / "train.txt").write_text("data/raw/div2k_train/images/0001.png\n")


def test_data_dir_works_on_a_fresh_clone_with_tracked_data_files(tmp_path):
    """The bug that filled /kaggle/working: data/ was not empty after the clone, so --data-dir was silently ignored."""
    root, big = tmp_path / "proj", tmp_path / "temp" / "sphota_data"
    _fresh_clone(root)
    try:
        msg = kaggle.prepare_data(root, data_dir=big)
    except OSError:
        pytest.skip("symlinks are not allowed on this machine")
    assert "data ->" in msg and (root / "data").is_symlink()
    assert (big / "processed" / "train.txt").read_text().startswith("data/raw/div2k_train")      # the tracked file was kept (merged)
    (root / "data" / "raw" / "kodak").mkdir()                                                    # writes through data/ land in the big folder
    (root / "data" / "raw" / "kodak" / "a.png").write_bytes(b"x")
    assert (big / "raw" / "kodak" / "a.png").exists()
    assert kaggle.prepare_data(root, data_dir=big).startswith("data ->")                          # running again is fine


def test_existing_files_in_the_big_folder_win_over_the_clone(tmp_path):
    root, big = tmp_path / "proj", tmp_path / "big"
    _fresh_clone(root)
    (big / "processed").mkdir(parents=True)
    (big / "processed" / "train.txt").write_text("restored\n")
    try:
        kaggle.prepare_data(root, data_dir=big)
    except OSError:
        pytest.skip("symlinks are not allowed on this machine")
    assert (big / "processed" / "train.txt").read_text() == "restored\n"


def test_raw_present_ignores_the_split_lists_and_placeholders(tmp_path):
    root = tmp_path / "proj"
    _fresh_clone(root)
    assert not kaggle.raw_present(root / "data")                              # lists + .gitkeep prove nothing
    (root / "data" / "raw" / "kodak").mkdir()
    assert not kaggle.raw_present(root / "data")
    (root / "data" / "raw" / "kodak" / "k.png").write_bytes(b"x")
    assert kaggle.raw_present(root / "data")


def test_dataset_restore_is_not_skipped_because_the_clone_contains_train_txt(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    _fresh_clone(root)
    calls = []
    monkeypatch.setattr(kaggle, "data_pull", lambda repo, d: calls.append(repo) or True)
    assert "restored from dataset u/d" in kaggle.prepare_data(root, data_repo="u/d")
    assert calls == ["u/d"]


def test_no_handover_to_the_gpu_when_the_data_notebook_downloaded_nothing(tmp_path):
    action, push = run_after(tmp_path, cfg(kind="cpu_data"), kaggle.DONE, last={"gpu": False, "hours_used": 0.1}, ready=False)
    assert action == "stop" and push.pushed == []
    action, push = run_after(tmp_path, cfg(kind="cpu_data"), kaggle.DONE, last={"gpu": False, "hours_used": 2.0}, ready=True)
    assert action == "gpu" and len(push.pushed) == 1


def test_dataset_restore_reports_failure_when_the_repo_has_no_images(tmp_path, monkeypatch):
    """huggingface_hub returns the local folder instead of raising when the repo does not exist: that is not a success."""
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    assert sync.data_pull("u/missing", tmp_path / "data") is False
    (tmp_path / "data" / "raw" / "kodak").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "kodak" / "k.png").write_bytes(b"x")
    assert sync.data_pull("u/there", tmp_path / "data") is True


# ------------------------------------------------------------------------------------------------ no duplicate GPU sessions
def test_active_sessions_are_the_unfinished_recently_refreshed_ones():
    now = 10_000.0
    e = [{"session": "a", "start": now - 3000, "reserved_h": 11, "hours_used": None, "last_seen": now - 60},     # running
         {"session": "b", "start": now - 9000, "reserved_h": 11, "hours_used": None, "last_seen": now - 4000},   # killed: not refreshed
         {"session": "c", "start": now - 3000, "reserved_h": 11, "hours_used": 0.8, "last_seen": now - 60},      # finished
         {"session": "d", "start": now - 100, "reserved_h": 11, "hours_used": None}]                            # old format: start counts
    assert [x["session"] for x in kaggle.active_sessions(e, now)] == ["a", "d"]


def test_a_second_gpu_session_stops_at_once_and_does_not_chain(tmp_path):
    exp, store = tmp_path / "exp", tmp_path / "store"
    exp.mkdir()
    kaggle.write_ledger_entry(exp, "other", kaggle.time.time() - 600, 11.5)                    # another session started 10 min ago, alive
    ran = tmp_path / "ran"
    status, rc = kaggle.run_session(ROOT, f"dir:{store}", hours=11.5, cmd=[sys.executable, "-c", f"open({str(ran)!r},'w')"], exp_dir=exp,
                                    do_git_pull=False, install=False, weekly_quota=30, gpu=True)
    assert (status, rc) == (kaggle.BUSY, kaggle.EXIT_BUSY) and not ran.exists()
    assert chain.decide("gpu", kaggle.BUSY, 0, 40, True)[0] == "stop" and kaggle.classify(77) == kaggle.BUSY
    assert len(kaggle.read_ledger(exp)) == 1                                                    # it wrote nothing of its own


def test_a_dead_session_does_not_block_and_a_running_one_refreshes_its_entry(tmp_path):
    exp, store = tmp_path / "exp", tmp_path / "store"
    exp.mkdir()
    kaggle.write_ledger_entry(exp, "dead", kaggle.time.time() - 7200, 11.5, last_seen=kaggle.time.time() - 3600)   # not refreshed for 1 h
    code = "import time; time.sleep(2.5)"
    status, rc = kaggle.run_session(ROOT, f"dir:{store}", hours=11.5, cmd=[sys.executable, "-c", code], exp_dir=exp, do_git_pull=False,
                                    install=False, weekly_quota=30, gpu=True, heartbeat_s=0.2)
    assert (status, rc) == (kaggle.DONE, 0)
    mine = [e for e in kaggle.read_ledger(exp) if e["session"] != "dead"][0]
    assert mine["hours_used"] is not None and mine["hours_used"] > 0                            # finished: no longer "active"


def test_job_console_logs_are_kept_in_the_run_repo():
    """A crashed job's traceback lives in experiments/logs/job_<name>.out: it must reach Hugging Face so it can be read after the session."""
    assert sync.wanted("logs/job_lmbda0.0483.out") and sync.wanted("logs/job_diffusion.out") and sync.wanted("logs/diffusion.log")


# ------------------------------------------------------------------------------------------------ rate limits (HTTP 429) and failed restores
def test_retry_hf_retries_a_rate_limit_and_gives_up_after_the_last_attempt():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("HTTP status client error (429 Too Many Requests)")
        return "ok"
    assert sync.retry_hf(flaky, "test", sleep=waits.append, log=lambda m, **k: None) == "ok" and len(calls) == 3
    assert waits == [30.0, 60.0]                                                       # the wait doubles
    always = lambda: (_ for _ in ()).throw(RuntimeError("429"))
    with pytest.raises(RuntimeError):
        sync.retry_hf(always, "test", attempts=3, sleep=lambda s: None, log=lambda m, **k: None)


def test_retry_hf_does_not_retry_a_missing_repo():
    n = []

    class Missing(Exception):
        pass

    def f():
        n.append(1)
        raise Missing()
    with pytest.raises(Missing):
        sync.retry_hf(f, "test", fatal=(Missing,), sleep=lambda s: None, log=lambda m, **k: None)
    assert len(n) == 1


def test_a_failed_restore_stops_the_session_instead_of_starting_from_scratch(tmp_path, monkeypatch):
    """The first upload of a session that carried on after a failed restore would replace the real checkpoints with fresh ones."""
    def boom(self, *a, **k):
        raise RuntimeError("429 Too Many Requests")
    monkeypatch.setattr(sync.Syncer, "pull", boom)
    ran = tmp_path / "ran"
    exp = tmp_path / "exp"; exp.mkdir()
    status, rc = kaggle.run_session(ROOT, f"dir:{tmp_path / 'store'}", hours=1, cmd=[sys.executable, "-c", f"open({str(ran)!r},'w')"], exp_dir=exp,
                                    do_git_pull=False, install=False, gpu=False)
    assert (status, rc) == (kaggle.CRASHED, 1) and not ran.exists() and not (tmp_path / "store").exists()


# ------------------------------------------------------------------------------------------------ a cancelled session must not count as a full one
def test_a_cancelled_session_is_settled_from_its_heartbeat_or_the_dead_hours_setting():
    now = 1_000_000_000.0
    day = 86400.0
    cancelled_new = {"start": now - day, "reserved_h": 11.5, "hours_used": None, "last_seen": now - day + 1500}      # ran 25 min, then died
    cancelled_old = {"start": now - day, "reserved_h": 11.5, "hours_used": None, "last_seen": now - day}              # no heartbeat evidence
    running = {"start": now - 600, "reserved_h": 11.5, "hours_used": None, "last_seen": now - 30}
    assert kaggle.used_quota_hours([cancelled_new], now) == pytest.approx(25 / 60 + 0.25, abs=0.01)                    # not 11.5 h
    assert kaggle.used_quota_hours([cancelled_old], now) == pytest.approx(11.5)                                        # unknown: stay safe
    assert kaggle.used_quota_hours([cancelled_old], now, dead_hours=0.5) == pytest.approx(0.5)                         # you said: assume 0.5 h
    assert kaggle.used_quota_hours([running], now, dead_hours=0.5) == pytest.approx(600 / 3600 * 1.0 + 0.0, abs=1.2)  # a live session still counts (reservation capped at now)
    b, _ = kaggle.plan_budget(11.5, weekly_quota=30, entries=[{"start": now - 3 * day, "reserved_h": 11.5, "hours_used": 11.4},
                                                            cancelled_old], now=now, dead_hours=0.5)
    assert b == pytest.approx(11.5)                                                                                      # 30 - 11.9 - 0.5 > 11.5
    b2, _ = kaggle.plan_budget(11.5, weekly_quota=30, entries=[{"start": now - 3 * day, "reserved_h": 11.5, "hours_used": 11.4},
                                                             cancelled_old], now=now)
    assert b2 == pytest.approx(30 - 11.4 - 11.5 - 0.5)                                                                   # without the setting: 6.6 h
