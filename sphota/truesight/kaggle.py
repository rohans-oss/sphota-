"""One Kaggle session of the full run: restore, run until the time budget, save, stop.   (driver: scripts/kaggle_run.py)

Kaggle stops a session after 12 hours and wipes its disk. A session therefore does exactly this:

    1. pick up the latest code           git pull   (so a bug you fixed on GitHub is used)
    2. make sure the packages exist      pip install only what is missing
    3. restore the data                  attached Kaggle dataset, or a Hugging Face dataset repo, or download once
    4. restore the run state             pull experiments/ (checkpoints, logs, stage markers) from storage
    5. run the pipeline                  every trainer resumes from its last checkpoint; stops itself before the 12 h limit
    6. save the run state                final push, plus a push every 30 minutes while running

Weekly GPU quota. Kaggle ends a session without warning when the weekly GPU quota is used up, and no API tells a notebook how
much is left. The driver therefore plans the budget itself, and never starts a session it cannot finish cleanly:
    --quota-hours R     R = the hours the Kaggle page shows as left right now:  budget = min(--hours, R - 0.5)
    --weekly-quota 30   the driver keeps a ledger (quota/<session>.json in the run state, uploaded with everything else) and
                        counts the hours of the last 7 days itself:  budget = min(--hours, 30 - used - 0.5).  A session that was
                        killed counts as having used its whole reservation, so the count can only be too high, never too low.
With a budget below 15 minutes the session does not start (status NO_QUOTA, exit 76). Because the trainers pause and save when the
budget is used up, the quota is never hit mid-run; the 0.5 h margin leaves time for the final upload. If the quota is wrong anyway
(for example Kaggle counted a session you forgot), the periodic checkpoints (every 9 min) and uploads (every 15 min) limit the loss
to about 15 minutes.

Outcomes (the exit status of the pipeline):
    DONE     0    every requested stage finished
    PAUSED   75   the time budget ran out (state saved): start the next session, it continues exactly there
    CRASHED  other  a bug or an out-of-memory error: the trainer already wrote its exact state before dying and this driver
                    uploaded it; fix the code (push the fix to GitHub), start the next session, it continues from that step
"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from . import runtime
from .sync import Syncer, data_pull, data_push, raw_present, release_all

DONE, PAUSED, CRASHED, NO_QUOTA, BUSY = "DONE", "PAUSED", "CRASHED", "NO_QUOTA", "BUSY"
EXIT_NO_QUOTA = 76                       # not enough weekly GPU quota left to run a useful session
EXIT_BUSY = 77                           # another GPU session of this run is alive: this one stops at once (no duplicate training)
ACTIVE_STALE_S = 45 * 60                 # a session whose ledger entry was not refreshed for this long is taken to be dead
DEFAULT_HOURS = 11.5                     # Kaggle stops at 12 h: the last 30 minutes are for the final upload
QUOTA_MARGIN_H = 0.5                     # kept free of the quota for the final upload and for estimate errors
MIN_USEFUL_H = 0.25                      # below this budget a session cannot even restore the state and save it again
QUOTA_WINDOW_H = 7 * 24                  # the ledger counts the last 7 days (a rolling week never under-counts a weekly reset)
LAST_SESSION = {}                        # filled by run_session: session id, start, budget_h, gpu, hours_used, status, rc
DEFAULT_SYNC_MIN = 15.0                  # upload interval; checkpoints are written more often (see session_env)
REQUIRED = {                              # import name -> pip name
    "compressai": "compressai", "torchac": "torchac", "lpips": "lpips", "diffusers": "diffusers", "easyocr": "easyocr",
    "facenet_pytorch": "facenet-pytorch", "sklearn": "scikit-learn", "xgboost": "xgboost", "mlxtend": "mlxtend",
    "cleanfid": "clean-fid", "editdistance": "editdistance", "imagehash": "imagehash", "pytorch_msssim": "pytorch-msssim",
    "huggingface_hub": "huggingface_hub", "pandas": "pandas", "matplotlib": "matplotlib", "scipy": "scipy", "yaml": "pyyaml",
    "cv2": "opencv-python", "ninja": "ninja", "joblib": "joblib", "requests": "requests", "pyarrow": "pyarrow",
}
NO_DEPS = {"facenet-pytorch"}             # its pins would replace the CUDA torch Kaggle ships with


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


SECRET_NAMES = ("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "GITHUB_TOKEN")   # missing ones are simply skipped


def has_gpu():
    """True when this machine has a CUDA GPU (a CPU-only session does not use the weekly GPU quota)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_kaggle_secrets(names=SECRET_NAMES):
    """Copy Kaggle 'Add-ons > Secrets' into environment variables (does nothing outside Kaggle, or with TS_NO_KAGGLE_SECRETS=1, which
    the test suite sets so that tests never pick up the real tokens of the notebook they run in)."""
    if os.environ.get("TS_NO_KAGGLE_SECRETS"):
        return []
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return []
    got = []
    for n in names:
        if os.environ.get(n):
            continue
        try:
            os.environ[n] = UserSecretsClient().get_secret(n); got.append(n)
        except Exception:
            pass
    return got


def missing_packages(required=REQUIRED):
    return [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]


def ensure_requirements(required=REQUIRED):
    miss = missing_packages(required)
    if not miss:
        log("all packages present")
        return []
    log(f"installing missing packages: {', '.join(miss)}")
    normal = [m for m in miss if m not in NO_DEPS]
    if normal:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *normal], check=True)
    for m in (m for m in miss if m in NO_DEPS):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", m], check=True)
    return miss


def git_pull(root):
    """Fast-forward to the latest commit so a fix pushed to GitHub is used. Never fails the session."""
    if not (Path(root) / ".git").exists():
        return "not a git checkout"
    r = subprocess.run(["git", "pull", "--ff-only"], cwd=root, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    return out[-1] if out else "ok"


def _only_placeholders(d):
    """True if the folder holds nothing but .gitkeep files (what a fresh git clone leaves in data/raw)."""
    return all(p.name == ".gitkeep" for p in Path(d).rglob("*") if p.is_file())


def link(src, dst, merge=False):
    """dst -> src (symlink). An empty folder (or one with only .gitkeep files) is replaced. A folder with real content is left
    alone, unless ``merge``: then its files that are missing in ``src`` are copied there first (a fresh clone of the repository
    already contains data/processed/*.txt and .gitkeep files, which must not stop the link), and the folder is replaced."""
    import shutil
    src, dst = Path(src), Path(dst)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        if _only_placeholders(dst):
            shutil.rmtree(dst)
        elif merge:
            src.mkdir(parents=True, exist_ok=True)
            for f in dst.rglob("*"):
                if f.is_file():
                    t = src / f.relative_to(dst)
                    if not t.exists():
                        t.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, t)
            shutil.rmtree(dst)
        else:
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)
    return True


def prepare_data(root, data_input=None, data_dir=None, data_repo=None):
    """Make data/raw and data/processed available. Returns a short description of what was done.

    data_input  folder of an attached Kaggle dataset that contains raw/ (and optionally processed/): linked, instant
    data_dir    a roomy scratch folder (Kaggle: /kaggle/temp/...) that becomes <root>/data, because /kaggle/working holds 20 GB
    data_repo   Hugging Face dataset repo uploaded earlier with ``python -m truesight.sync data-push``: downloaded if the raw images are missing
    With none of them the pipeline's own ``data`` stage downloads the datasets from their original sources (slow, once)."""
    root, data = Path(root), Path(root) / "data"
    notes = []
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        if link(data_dir, data, merge=True):
            import shutil
            free = shutil.disk_usage(data_dir).free / 2 ** 30
            notes.append(f"data -> {data_dir} ({free:.0f} GB free there)" + ("  WARNING: less than ~30 GB free: the datasets need about 23 GB, "
                                                                           "the image cache up to 41 GB more" if free < 30 else ""))
        else:
            raise RuntimeError(f"could not link {data} to {data_dir}: it is a real folder with files and could not be merged")
    if data_input and (Path(data_input) / "raw").exists():
        (data).mkdir(parents=True, exist_ok=True)
        if link(Path(data_input) / "raw", data / "raw"):
            notes.append(f"data/raw -> {data_input}/raw")
        if (Path(data_input) / "processed").exists() and not (data / "processed" / "train.txt").exists():
            import shutil
            shutil.copytree(Path(data_input) / "processed", data / "processed", dirs_exist_ok=True)
            notes.append("processed lists copied")
    elif data_repo and not raw_present(data):
        if data_pull(data_repo, data):
            notes.append(f"restored from dataset {data_repo}")
    return "; ".join(notes) or "no data source given: the data stage downloads from the original sources"


def session_env(spec, hours, start=None, data_repo=None, base=None, sync_every=DEFAULT_SYNC_MIN, session=None):
    env = dict(base if base is not None else os.environ)
    env["TS_SESSION_START"] = str(start if start is not None else time.time())
    env["TS_TIME_LIMIT_HOURS"] = str(hours)
    env["TS_SYNC"] = spec
    env["TS_SAVE_EVERY_MIN"] = f"{max(2.0, min(20.0, 0.6 * sync_every)):.1f}"      # a checkpoint exists before every upload
    env["TS_CLAIM_TTL_MIN"] = f"{max(45.0, 3 * sync_every):.0f}"
    if session:
        env["TS_SESSION_ID"] = session
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if data_repo:
        env["TS_DATA_REPO"] = data_repo
    return env


def classify(rc):
    return DONE if rc == 0 else PAUSED if rc == runtime.EXIT_PAUSED else NO_QUOTA if rc == EXIT_NO_QUOTA else BUSY if rc == EXIT_BUSY else CRASHED


# ------------------------------------------------------------------------------------------------ weekly quota ledger
def ledger_dir(exp):
    return Path(exp) / "quota"


def read_ledger(exp):
    """All session records of the run state: list of dicts {session, start, reserved_h, hours_used (None while running / killed)}."""
    out = []
    for f in sorted(ledger_dir(exp).glob("*.json")) if ledger_dir(exp).exists() else []:
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    return out


def write_ledger_entry(exp, session, start, reserved_h, hours_used=None, last_seen=None):
    d = ledger_dir(exp)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session}.json.tmp"
    tmp.write_text(json.dumps({"session": session, "start": start, "reserved_h": reserved_h, "hours_used": hours_used,
                               "last_seen": time.time() if last_seen is None else last_seen}))
    os.replace(tmp, d / f"{session}.json")


def active_sessions(entries, now=None, stale_s=ACTIVE_STALE_S):
    """Ledger entries of GPU sessions that look alive: not finished (no ``hours_used``) and refreshed (``last_seen``) recently. A running
    session refreshes its entry every minute and uploads it with every sync, so a killed one is taken as dead after ``stale_s``."""
    now = time.time() if now is None else now
    return [e for e in entries if e.get("hours_used") is None and now - float(e.get("last_seen", e.get("start", 0))) < stale_s]


def last_reset(now, dow, hour=0):
    """Epoch seconds of the most recent weekly reset at (weekday ``dow`` with Monday = 0, UTC ``hour``) at or before ``now``."""
    import datetime as dt
    t = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    r = t.replace(hour=int(hour), minute=0, second=0, microsecond=0) - dt.timedelta(days=(t.weekday() - int(dow)) % 7)
    if r > t:
        r -= dt.timedelta(days=7)
    return r.timestamp()


def next_reset(now, dow, hour=0):
    """Epoch seconds of the next weekly reset strictly after ``now`` (same convention as ``last_reset``)."""
    return last_reset(now, dow, hour) + 7 * 86400.0


def used_quota_hours(entries, now=None, window_h=QUOTA_WINDOW_H, since=None, dead_hours=None):
    """GPU-session hours spent in the last ``window_h`` hours (or since the epoch time ``since``, the last weekly reset, when given).
    A finished session counts its real hours, a session without ``hours_used`` (still running or killed) counts its whole
    reservation; only the part inside the window counts.

    A session that is no longer alive but never wrote its end (cancelled, killed) is settled from the best evidence: its last heartbeat
    (``last_seen``, refreshed every minute while it ran) if there is one; else ``dead_hours`` if you gave it (--dead-session-hours:
    "assume a dead session used this many hours"); else, to be safe, its whole reservation. A session still alive counts in full."""
    now = time.time() if now is None else now
    lo, total = (now - window_h * 3600.0) if since is None else since, 0.0
    for e in entries:
        h = e.get("hours_used")
        if h is None:
            ls, start = e.get("last_seen"), float(e["start"])
            alive = ls is not None and now - float(ls) < ACTIVE_STALE_S
            if not alive and ls is not None and float(ls) - start > 90:             # a heartbeat proves how long it ran
                h = min(float(e["reserved_h"]), (float(ls) - start) / 3600.0 + 0.25)
            elif not alive and dead_hours is not None:
                h = min(float(e["reserved_h"]), float(dead_hours))
        h = float(e["reserved_h"] if h is None else h)
        a, b = float(e["start"]), float(e["start"]) + h * 3600.0
        overlap = max(0.0, min(b, now) - max(a, lo))
        total += overlap / 3600.0
    return total


def plan_budget(hours, quota_hours=None, weekly_quota=None, entries=(), now=None, margin_h=QUOTA_MARGIN_H, reset=None, dead_hours=None):
    """(budget in hours, explanation). ``hours`` is the session limit (11.5); the quota only ever lowers it.
    ``reset`` = (weekday, utc_hour) of the weekly quota reset (Monday = 0): the ledger then counts only the hours since the last
    reset instead of the last 7 days (exact, and the quota is full again right after the reset)."""
    budget, why = float(hours), f"session limit {hours} h"
    if quota_hours is not None and quota_hours - margin_h < budget:
        budget, why = quota_hours - margin_h, f"{quota_hours} h of quota left - {margin_h} h margin"
    if weekly_quota is not None:
        t = time.time() if now is None else now
        used = used_quota_hours(entries, t, since=None if reset is None else last_reset(t, *reset), dead_hours=dead_hours)
        left = weekly_quota - used - margin_h
        if left < budget:
            span = "in the last 7 days" if reset is None else "since the last weekly reset"
            budget, why = left, f"weekly quota {weekly_quota} h - {used:.1f} h used {span} - {margin_h} h margin"
    return max(0.0, budget), why


def run_session(root, spec, hours=DEFAULT_HOURS, cmd=None, exp_dir=None, data_input=None, data_dir=None, data_repo=None,
                sync_every=DEFAULT_SYNC_MIN, do_git_pull=True, install=True, env_extra=None, quota_hours=None, weekly_quota=None,
                gpu=None, reset=None, heartbeat_s=60.0, dead_hours=None):
    """Run one session. Returns (status, exit code). ``cmd`` defaults to the full pipeline.
    ``quota_hours`` / ``weekly_quota`` / ``reset``: see the module docstring; the budget is lowered to what the quota allows.
    ``gpu``: does this session use the GPU quota? None = detect. A CPU-only session (the data download) uses no quota: it gets the full
    session limit and writes no ledger entry. ``LAST_SESSION`` describes the session afterwards (used by chain.py)."""
    LAST_SESSION.clear()
    root = Path(root)
    exp = Path(exp_dir) if exp_dir else root / "experiments"
    start = time.time()
    session = f"{int(start)}-{uuid.uuid4().hex[:6]}"
    got = load_kaggle_secrets()
    log(f"session start; budget {hours} h; run state in {spec}" + (f"; secrets loaded: {got}" if got else ""))
    if do_git_pull:
        log(f"git pull: {git_pull(root)}")
    if install:
        ensure_requirements()
    log(f"data: {prepare_data(root, data_input, data_dir, data_repo)}")
    sync = Syncer(spec, exp)
    try:
        sync.pull()                                                # (a repo that does not exist yet, the first session, is handled inside)
    except Exception as e:
        # NEVER carry on from scratch after a failed restore: the first upload would replace your real checkpoints with fresh ones
        log(f"CANNOT RESTORE THE RUN STATE ({type(e).__name__}: {e}). Stopping so that nothing in {spec} is overwritten. Run again.")
        return CRASHED, 1
    hours_limit = hours
    gpu = has_gpu() if gpu is None else gpu
    if gpu:
        hours, why = plan_budget(hours_limit, quota_hours, weekly_quota, read_ledger(exp), start, reset=reset, dead_hours=dead_hours)
    else:
        why = "CPU-only session: no GPU quota used"
    LAST_SESSION.update(session=session, start=start, budget_h=hours, gpu=gpu, hours_used=0.0)
    if gpu:
        alive = active_sessions(read_ledger(exp), start)
        if alive:                                                   # two sessions on one run would train the same stages twice
            log(f"BUSY: another GPU session of this run is running (started {time.strftime('%H:%M UTC', time.gmtime(alive[0]['start']))}, "
                f"refreshed {int(start - float(alive[0].get('last_seen', alive[0]['start']))) // 60} min ago): this one stops at once so nothing is "
                f"trained twice and no quota is wasted. If that session is really dead, run again in {ACTIVE_STALE_S // 60} minutes.")
            return BUSY, EXIT_BUSY
    if gpu and quota_hours is None and weekly_quota is None:
        log("WARNING: no quota given (--quota-hours / --weekly-quota). Kaggle ends a session without warning when the weekly GPU "
            "quota runs out and this driver cannot see it; pass the hours the Kaggle page shows as left.")
    log(f"time budget {hours:.2f} h ({why})")
    if hours < MIN_USEFUL_H and hours < hours_limit:               # the quota (not a short test limit) left too little
        log(f"NO_QUOTA: only {hours:.2f} h of budget left, not starting (the run state is untouched). Try again when the weekly quota resets.")
        return NO_QUOTA, EXIT_NO_QUOTA
    done_before = sorted(p.name for p in (exp / ".done").glob("*")) if (exp / ".done").exists() else []
    log(f"stages already finished: {done_before or 'none'}")
    stop_beat = threading.Event()
    if gpu:
        write_ledger_entry(exp, session, start, hours)              # reserve it now: a killed session still counts

        def beat():                                                 # "I am alive": refreshed every minute, uploaded with every sync
            while not stop_beat.wait(heartbeat_s):
                try:
                    write_ledger_entry(exp, session, start, hours)
                except OSError:
                    pass
        threading.Thread(target=beat, daemon=True).start()
    sync.session = session
    try:
        sync.push()
    except Exception as e:
        log(f"ledger upload failed (the next push retries): {e}")
    sync.start_background(sync_every)
    env = session_env(spec, hours, start, data_repo, sync_every=sync_every, session=session)
    env.update(env_extra or {})
    cmd = cmd or ["bash", str(root / "scripts" / "run_full_pipeline.sh")]
    log("running: " + " ".join(map(str, cmd)))
    rc = subprocess.run(cmd, cwd=root, env=env).returncode
    status = classify(rc)
    sync.stop_background()
    stop_beat.set()
    used = round((time.time() - start) / 3600.0, 4)
    LAST_SESSION.update(hours_used=used, status=status, rc=rc)
    if gpu:
        write_ledger_entry(exp, session, start, hours, hours_used=used)
    release_all(exp, session)                                       # the next session may take over this session's stages at once
    log(f"pipeline ended with status {status} (exit {rc}); saving the run state")
    for attempt in range(3):
        try:
            sync.push(squash=status != CRASHED)
            break
        except Exception as e:
            log(f"final push failed ({type(e).__name__}: {e}); retrying" if attempt < 2 else
                f"FINAL PUSH FAILED: {e}. The state is still in {exp}; if this is a Kaggle notebook, save the version so /kaggle/working is kept.")
            time.sleep(5 * (attempt + 1))
    if data_repo and raw_present(root / "data") and not (exp / ".done" / "data_uploaded").exists() \
            and not (root / "data" / "raw").is_symlink() and not (Path(data_input or "x") / "raw").exists():
        try:                                                       # first session: keep the 23 GB so later sessions restore in minutes
            data_push(data_repo, root / "data")
            (exp / ".done").mkdir(parents=True, exist_ok=True); (exp / ".done" / "data_uploaded").write_text("")
            sync.push()
        except Exception as e:
            log(f"data upload failed (not fatal, the next session retries): {e}")
    minutes = (time.time() - start) / 60
    log({DONE: "ALL STAGES FINISHED.",
         PAUSED: f"PAUSED after {minutes:.0f} min: the state is saved. Start the next session (run the same cell); it continues from here.",
         CRASHED: "CRASHED. The exact state was saved before the failure and has been uploaded. Read the error above, fix the code "
                  "(git push the fix), then start the next session: it continues from that step, not from the beginning."}[status])
    return status, rc
