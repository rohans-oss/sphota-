"""Parallel jobs on several GPUs, the shell chain runner, overwrite-safe syncing with stage claims, and the quota-aware budget
(truesight/parallel.py, scripts/chains.sh, truesight/sync.py, truesight/kaggle.py)."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from truesight import kaggle, parallel, sync
from truesight.sync import Syncer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# ------------------------------------------------------------------------------------------------ parallel jobs
def _py(code):
    return [sys.executable, "-c", code]


def _marker(path, body="", sleep=0.0, rc=0):
    """A job that records when it started / ended and the GPU it was pinned to, then exits with ``rc``."""
    return _py(f"import os,time,pathlib; p=pathlib.Path({str(path)!r}); p.write_text('start %s gpu=%s\\n' % (time.time(), os.environ['CUDA_VISIBLE_DEVICES']));"
               f"time.sleep({sleep}); p.write_text(p.read_text()+'end %s\\n' % time.time()); raise SystemExit({rc})")


def _times(path):
    lines = Path(path).read_text().splitlines()
    return float(lines[0].split()[1]), float(lines[1].split()[1]), lines[0].split("gpu=")[1]


def test_independent_jobs_overlap_on_different_gpus_and_dependents_wait(tmp_path):
    jobs = [parallel.Job("base", _marker(tmp_path / "base", sleep=0.6)),
            parallel.Job("ft1", _marker(tmp_path / "ft1", sleep=0.6), ["base"]),
            parallel.Job("ft2", _marker(tmp_path / "ft2", sleep=0.6), ["base"]),
            parallel.Job("ft3", _marker(tmp_path / "ft3", sleep=0.1), ["base"])]
    assert parallel.run_jobs(jobs, gpus=["0", "1"], poll=0.05, log=lambda m: None) == 0
    b0, b1, _ = _times(tmp_path / "base")
    f1, f2 = _times(tmp_path / "ft1"), _times(tmp_path / "ft2")
    assert f1[0] >= b1 and f2[0] >= b1                              # fine-tunes never start before their base model has finished
    assert f1[0] < f2[1] and f2[0] < f1[1]                          # ... but they run at the same time
    assert {f1[2], f2[2]} == {"0", "1"}                             # one per GPU
    assert _times(tmp_path / "ft3")[0] >= min(f1[1], f2[1])         # a third job waits for a free GPU


def test_one_gpu_is_strictly_serial(tmp_path):
    jobs = [parallel.Job(n, _marker(tmp_path / n, sleep=0.2)) for n in "abc"]
    assert parallel.run_jobs(jobs, gpus=["0"], poll=0.05, log=lambda m: None) == 0
    t = [_times(tmp_path / n) for n in "abc"]
    assert t[0][1] <= t[1][0] and t[1][1] <= t[2][0]


def test_finished_jobs_are_skipped(tmp_path):
    (tmp_path / "final.pt").write_text("x")
    jobs = [parallel.Job("a", _py("raise SystemExit(9)"), skip_if=tmp_path / "final.pt"), parallel.Job("b", _py("pass"), ["a"])]
    assert parallel.run_jobs(jobs, gpus=["0"], poll=0.05, log=lambda m: None) == 0


def test_a_crash_stops_new_jobs_and_is_reported_over_a_pause(tmp_path):
    jobs = [parallel.Job("bad", _marker(tmp_path / "bad", sleep=0.2, rc=3)),
            parallel.Job("pause", _marker(tmp_path / "pause", sleep=0.4, rc=75)),
            parallel.Job("later", _marker(tmp_path / "later"), ["bad"])]
    assert parallel.run_jobs(jobs, gpus=["0", "1"], poll=0.05, log=lambda m: None) == 3
    assert not (tmp_path / "later").exists()                        # nothing starts after a failure


def test_a_pause_returns_75(tmp_path):
    jobs = [parallel.Job("p", _marker(tmp_path / "p", rc=75)), parallel.Job("q", _marker(tmp_path / "q"), ["p"])]
    assert parallel.run_jobs(jobs, gpus=["0"], poll=0.05, log=lambda m: None) == 75
    assert not (tmp_path / "q").exists()


def test_bad_job_graph_is_rejected():
    with pytest.raises(ValueError):
        parallel.run_jobs([parallel.Job("a", ["x"], ["missing"])], gpus=["0"])
    with pytest.raises(ValueError):
        parallel.run_jobs([parallel.Job("a", ["x"]), parallel.Job("a", ["y"])], gpus=["0"])


def test_visible_gpus_follows_the_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert parallel.visible_gpus() == ["2", "3"]


def test_codec_jobs_keep_the_bases_first_and_fine_tunes_after_them(tmp_path):
    import train_all_lambdas as t
    a = argparse.Namespace(base_iters=100, finetune_iters=10, N=16, M=24, batch=2, crop=64, workers=0, ckpt_every=50,
                           train_list=None, from_scratch=False)
    jobs = t.build_jobs(tmp_path, [(3, "gdn", list(t.LAMBDAS))], a)
    by = {j.name: j for j in jobs}
    assert by["lmbda0.0130"].after == [] and "--init-from" not in by["lmbda0.0130"].cmd
    assert jobs[1].name == "lmbda0.0018" and jobs[2].name == "lmbda0.0483"       # the diffusion and ROI codecs come first
    for n, j in by.items():
        if n != "lmbda0.0130":
            assert j.after == ["lmbda0.0130"] and "--init-from" in j.cmd
    ab = t.build_jobs(tmp_path, [(1, "gdn", [0.013, 0.0035]), (3, "relu", [0.013, 0.0035])], a)
    bases = [j for j in ab if not j.after]
    assert {j.name for j in bases} == {"K1_lmbda0.0130", "relu_lmbda0.0130"}      # the two ablation chains are independent


# ------------------------------------------------------------------------------------------------ shell chains
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_run_chains_parallel_failure_propagation_and_serial_fallback(tmp_path):
    script = f"""
set -euo pipefail
cd {ROOT.as_posix()!r}
source scripts/chains.sh
a() {{ echo "a:$CUDA_VISIBLE_DEVICES"; sleep 1; }}
b() {{ echo "b:$CUDA_VISIBLE_DEVICES"; sleep 1; }}
bad() {{ sleep 1; exit 3; }}
slow() {{ trap 'echo slow-got-term; exit 75' TERM; for i in $(seq 1 15); do sleep 1; done; echo slow-finished; }}
export CHAIN_POLL=0.2
NGPU=2 run_chains a b
r=0; NGPU=2 run_chains bad slow || r=$?; echo "rc=$r"
"""
    t0 = time.time()
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    out = r.stdout
    assert "a:0" in out and "b:1" in out                            # one chain per GPU
    assert "rc=3" in out                                            # the crash is what is reported ...
    assert "slow-got-term" in out and "slow-finished" not in out    # ... and the other chain was told to save and stop
    assert time.time() - t0 < 25


# ------------------------------------------------------------------------------------------------ sync: no overwriting
def test_push_sends_only_files_this_machine_changed(tmp_path):
    store = tmp_path / "store"
    (store / "run").mkdir(parents=True)
    (store / "run" / "last.pt").write_bytes(b"NEWER-from-session-A")
    b = tmp_path / "B"
    sb = Syncer(f"dir:{store}", b, session="B")
    sb.pull()                                                        # session B starts and sees A's checkpoint
    (store / "run" / "last.pt").write_bytes(b"EVEN-NEWER-from-session-A")   # A advances it afterwards
    (b / "other").mkdir()
    (b / "other" / "log.csv").write_text("x\n")
    assert sb.push() == 1                                            # only B's own new file
    assert (store / "run" / "last.pt").read_bytes() == b"EVEN-NEWER-from-session-A"   # B's stale copy did not overwrite it


def test_two_sessions_on_different_stages_never_touch_each_others_files(tmp_path):
    store = tmp_path / "store"
    a, b = tmp_path / "A", tmp_path / "B"
    sa, sb = Syncer(f"dir:{store}", a, session="A"), Syncer(f"dir:{store}", b, session="B")
    sa.pull(); sb.pull()
    (a / "step10").mkdir(); (a / "step10" / "last.pt").write_bytes(b"diffusion")
    (b / "step7").mkdir(); (b / "step7" / "last.pt").write_bytes(b"codec")
    sa.push(); sb.push()
    sa.push(); sb.push()
    assert (store / "step10" / "last.pt").read_bytes() == b"diffusion" and (store / "step7" / "last.pt").read_bytes() == b"codec"


def test_claims_block_a_second_session_until_released_or_expired(tmp_path):
    store = tmp_path / "store"
    sa, sb = Syncer(f"dir:{store}", tmp_path / "A", session="A"), Syncer(f"dir:{store}", tmp_path / "B", session="B")
    assert sa.try_claim("diffusion") is True
    assert sb.try_claim("diffusion") is False                        # A is running it
    assert sb.try_claim("codec") is True                             # another stage is free
    sync.release_all(tmp_path / "A", "A"); sa.push()
    assert sb.try_claim("diffusion") is True                         # A released it (paused / finished its session)
    # a claim that is not refreshed any more (the owner was killed) can be taken over
    sync._write_claim(store, "protect", "B", now=time.time() - 3 * 3600)
    assert sa.try_claim("protect", ttl_min=45) is True


def test_finished_stage_is_never_claimed_again(tmp_path):
    store = tmp_path / "store"
    (store / ".done").mkdir(parents=True)
    (store / ".done" / "codec").write_text("")
    assert Syncer(f"dir:{store}", tmp_path / "A", session="A").try_claim("codec") is False


def test_heartbeat_keeps_a_live_claim_alive(tmp_path):
    loc = tmp_path / "A"
    sync._write_claim(loc, "diffusion", "A", now=time.time() - 3600)
    sync.heartbeat(loc, "A")
    assert sync.claimed_by_other(loc, "diffusion", "B", ttl_min=45) is True


def test_claim_cli_exit_codes(tmp_path):
    store, env = tmp_path / "store", dict(os.environ, TS_SESSION_ID="S1")
    cmd = [sys.executable, "-m", "truesight.sync", "claim", "--stage", "ae", "--dir", str(tmp_path / "e"), "--spec", f"dir:{store}"]
    assert subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True).returncode == 0
    assert subprocess.run(cmd, cwd=ROOT, env=dict(env, TS_SESSION_ID="S2"), capture_output=True).returncode == 3


def test_file_lock_serialises_and_recovers_a_stale_lock(tmp_path):
    lock = tmp_path / "x.lock"
    with sync.file_lock(lock):
        with pytest.raises(TimeoutError):
            with sync.file_lock(lock, timeout_s=0.5):
                pass
    lock.write_text("")
    os.utime(lock, (time.time() - 7200, time.time() - 7200))        # left behind by a dead process
    with sync.file_lock(lock, timeout_s=2):
        pass


# ------------------------------------------------------------------------------------------------ quota-aware budget
def test_plan_budget_uses_the_smallest_of_session_limit_and_quota():
    assert kaggle.plan_budget(11.5)[0] == 11.5
    assert kaggle.plan_budget(11.5, quota_hours=30)[0] == 11.5
    assert kaggle.plan_budget(11.5, quota_hours=5.0)[0] == pytest.approx(4.5)       # 0.5 h kept free for the final upload
    assert kaggle.plan_budget(11.5, quota_hours=0.3)[0] == 0.0


def test_ledger_counts_the_last_week_and_a_killed_session_counts_in_full():
    now = 1_000_000_000.0
    e = [{"start": now - 3600 * 24, "reserved_h": 11.5, "hours_used": 10.0},      # finished: its real hours
         {"start": now - 3600 * 5, "reserved_h": 4.0, "hours_used": None},        # killed / still running: the reservation
         {"start": now - 3600 * 24 * 20, "reserved_h": 11.5, "hours_used": 11.5}]  # too old: not counted
    assert kaggle.used_quota_hours(e, now) == pytest.approx(14.0)
    b, why = kaggle.plan_budget(11.5, weekly_quota=20, entries=e, now=now)
    assert b == pytest.approx(20 - 14.0 - 0.5) and "weekly quota" in why
    assert kaggle.plan_budget(11.5, weekly_quota=14.2, entries=e, now=now)[0] == 0.0          # nothing left: do not start


def test_a_session_with_no_quota_left_does_not_start_and_touches_nothing(tmp_path):
    exp, store = tmp_path / "exp", tmp_path / "store"
    exp.mkdir()
    kaggle.write_ledger_entry(exp, "old", time.time() - 3600 * 40, 30.0, last_seen=time.time() - 3600 * 40)   # a killed 30 h session: the week is used up
    called = tmp_path / "ran"
    status, rc = kaggle.run_session(ROOT, f"dir:{tmp_path / 'store'}", hours=11.5, cmd=_py(f"open({str(called)!r},'w')"), exp_dir=exp,
                                    do_git_pull=False, install=False, weekly_quota=30, gpu=True)
    assert (status, rc) == (kaggle.NO_QUOTA, kaggle.EXIT_NO_QUOTA) and not called.exists()


def test_session_lowers_its_budget_to_the_quota_and_writes_the_ledger(tmp_path):
    exp, store = tmp_path / "exp", tmp_path / "store"
    exp.mkdir()
    code = "import os; print('LIMIT', os.environ['TS_TIME_LIMIT_HOURS'], os.environ['TS_SAVE_EVERY_MIN'])"
    status, rc = kaggle.run_session(ROOT, f"dir:{store}", hours=11.5, cmd=_py(code), exp_dir=exp, do_git_pull=False, install=False,
                                    quota_hours=3.0, sync_every=15, gpu=True)
    assert (status, rc) == (kaggle.DONE, 0)
    (entry,) = kaggle.read_ledger(exp)
    assert entry["reserved_h"] == pytest.approx(2.5) and entry["hours_used"] is not None and entry["hours_used"] < 0.1
    assert any((store / "quota").glob("*.json"))                                    # the ledger is part of the uploaded run state


def test_session_env_checkpoints_before_every_upload():
    env = kaggle.session_env("hf:u/r", 4.5, start=1.0, base={}, sync_every=15, session="S")
    assert float(env["TS_SAVE_EVERY_MIN"]) < 15 and env["TS_SESSION_ID"] == "S" and env["TS_TIME_LIMIT_HOURS"] == "4.5"
    assert kaggle.classify(76) == kaggle.NO_QUOTA


# ------------------------------------------------------------------------------------------------ a pause must not become a "crash"
def test_a_pause_does_not_terminate_the_other_jobs(tmp_path):
    """One job pauses (75) first: the others keep running until they pause and save themselves (the real session budget is shared)."""
    slow = _py(f"import time,pathlib; time.sleep(1.0); pathlib.Path({str(tmp_path / 'saved')!r}).write_text('saved'); raise SystemExit(75)")
    jobs = [parallel.Job("a", _marker(tmp_path / "a", rc=75)), parallel.Job("b", slow)]
    assert parallel.run_jobs(jobs, gpus=["0", "1"], poll=0.05, log=lambda m: None) == 75
    assert (tmp_path / "saved").read_text() == "saved"                              # b was not killed mid-save


def test_a_job_stopped_by_the_runner_after_a_crash_is_not_a_second_crash(tmp_path):
    jobs = [parallel.Job("bad", _marker(tmp_path / "bad", sleep=0.2, rc=3)),
            parallel.Job("long", _py("import time; time.sleep(30)"))]
    t0 = time.time()
    assert parallel.run_jobs(jobs, gpus=["0", "1"], poll=0.05, log=lambda m: None) == 3     # the crash is reported, not 143
    assert time.time() - t0 < 20


@pytest.mark.skipif(not parallel.POSIX, reason="process groups exist only on POSIX")
def test_the_runner_waits_for_the_python_child_of_a_script_that_already_exited(tmp_path):
    """A shell wrapper that exits (e.g. after SIGTERM) while its child is still saving a checkpoint must not end the job early."""
    out = tmp_path / "child_done"
    wrapper = ["bash", "-c", f"(sleep 1.2; echo saved > {out.as_posix()}) & exit 0"]
    assert parallel.run_jobs([parallel.Job("w", wrapper)], gpus=["0"], poll=0.05, log=lambda m: None) == 0
    assert out.exists()
