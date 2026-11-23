"""Keep experiments/ (checkpoints, logs, stage markers) in persistent storage so a new Kaggle session can continue.

A Kaggle session has a 12-hour limit and its disk is wiped afterwards, so the run state lives outside it:

    hf:<user>/<repo>     a private Hugging Face model repo (needs the HF_TOKEN environment variable / Kaggle secret)
    dir:<path>           any directory, for example an attached Kaggle Dataset copy or a Google Drive mount

    python -m truesight.sync pull   --dir experiments --spec hf:user/sphota-run
    python -m truesight.sync push   --dir experiments --spec hf:user/sphota-run
    python -m truesight.sync data-push --data-repo user/sphota-data      # once: upload data/raw + split lists
    python -m truesight.sync data-pull --data-repo user/sphota-data      # every new session: restore them in minutes

Only run state is stored: *.pt, *.csv, *.json, *.png, *.npz, *.log, *.txt, *.md and the .done/ stage markers (no *.tmp files, nothing
larger than ``MAX_FILE_GB``). ``Syncer.start_background`` pushes every few minutes while training, so even a hard kill loses
little. Pull only fetches files that are missing or older locally; push only sends files that changed.

Safety when several processes or notebooks share one run repo (a T4 x2 session running two chains, or two notebooks):
  * a push sends ONLY the files this machine changed since its last pull / push (``.sync_baseline.json``), never a stale copy of
    a file that another session has advanced, so two sessions cannot overwrite each other's checkpoints;
  * pushes and pulls take a file lock, so two processes of one machine never commit at the same moment;
  * a stage is *claimed* before it starts (``.running/<stage>.json``, refreshed by every push). A claim that is not refreshed
    for ``CLAIM_TTL_MIN`` minutes (the owner died) can be taken over; a stage that is claimed by a live session, or finished
    (``.done/<stage>``), is skipped:   python -m truesight.sync claim --stage diffusion --dir experiments
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

KEEP_SUFFIXES = {".pt", ".csv", ".json", ".png", ".log", ".out", ".txt", ".md", ".yaml", ".npz"}   # .out = the console output of each job (experiments/logs/job_*.out)
MAX_FILE_GB = 4.0
BASELINE = ".sync_baseline.json"           # what this machine last saw in the storage (never uploaded itself)
LOCK = ".sync.lock"
CLAIM_DIR = ".running"
CLAIM_TTL_MIN = 45.0                        # a claim not refreshed for this long belongs to a dead session


def wanted(rel, size=None):
    """Should this path (relative to the run folder, with / separators) be persisted?"""
    p = Path(rel)
    if p.name.endswith(".tmp") or ".tmp" in p.suffixes:
        return False
    if any(part in ("__pycache__", ".cache", "fid") for part in p.parts):         # fid/ = rebuilt PNG folders, thousands of files
        return False
    if size is not None and size > MAX_FILE_GB * 2 ** 30:
        return False
    if p.name == ".stop_chain":                  # the kill switch of truesight/chain.py: an empty file in the run repo
        return True
    if p.name in (BASELINE, LOCK):
        return False
    if p.parts and p.parts[0] in (".done", CLAIM_DIR):
        return True
    return p.suffix.lower() in KEEP_SUFFIXES


def local_files(root):
    root = Path(root)
    out = {}
    if not root.exists():
        return out
    for f in root.rglob("*"):
        if f.is_file():
            rel = f.relative_to(root).as_posix()
            if wanted(rel, f.stat().st_size):
                out[rel] = f
    return out


def _stat_key(f):
    st = Path(f).stat()
    return [st.st_size, st.st_mtime_ns]


def load_baseline(local):
    """{relative path: [size, mtime_ns]} this machine last saw in the storage, or None if it never pulled / pushed."""
    try:
        return json.loads((Path(local) / BASELINE).read_text())
    except (OSError, ValueError):
        return None


def save_baseline(local, baseline):
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    tmp = local / (BASELINE + ".tmp")
    tmp.write_text(json.dumps(baseline))
    os.replace(tmp, local / BASELINE)


def changed_files(local, baseline):
    """Files to upload: those that are new or differ from what this machine last saw. ``baseline=None`` = everything."""
    out = {}
    for rel, f in local_files(local).items():
        if baseline is None or baseline.get(rel) != _stat_key(f):
            out[rel] = f
    return out


@contextmanager
def file_lock(path, timeout_s=3600.0, stale_s=3600.0):
    """Cross-process lock (a file created with O_EXCL). A lock older than ``stale_s`` is taken to belong to a dead process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_s:
                    path.unlink()
                    continue
            except OSError:
                continue
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"could not get the sync lock {path} in {timeout_s:.0f} s")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------- stage claims
def new_session_id():
    return os.environ.get("TS_SESSION_ID") or f"{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _claim_path(local, stage):
    return Path(local) / CLAIM_DIR / f"{stage}.json"


def _read_claim(local, stage):
    try:
        return json.loads(_claim_path(local, stage).read_text())
    except (OSError, ValueError):
        return None


def _write_claim(local, stage, session, released=False, now=None):
    p = _claim_path(local, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"session": session, "t": now if now is not None else time.time(), "released": released}))
    os.replace(tmp, p)


def claimed_by_other(local, stage, session, ttl_min=CLAIM_TTL_MIN, now=None):
    """True when another session holds a live (unreleased, recently refreshed) claim on ``stage``."""
    c = _read_claim(local, stage)
    if not c or c.get("released") or c.get("session") == session:
        return False
    return (now if now is not None else time.time()) - float(c.get("t", 0)) < ttl_min * 60


def claim_stage(local, stage, session, ttl_min=CLAIM_TTL_MIN, now=None):
    """Take the claim on ``stage`` unless the stage is finished or another live session owns it. Returns True if taken."""
    if (Path(local) / ".done" / stage).exists() or claimed_by_other(local, stage, session, ttl_min, now):
        return False
    _write_claim(local, stage, session, now=now)
    return True


def release_stage(local, stage, session):
    c = _read_claim(local, stage)
    if c and c.get("session") == session:
        _write_claim(local, stage, session, released=True)


def release_all(local, session):
    d = Path(local) / CLAIM_DIR
    for p in d.glob("*.json") if d.exists() else []:
        release_stage(local, p.stem, session)


def heartbeat(local, session, now=None):
    """Refresh the timestamp of this session's live claims (called before every push)."""
    d = Path(local) / CLAIM_DIR
    for p in d.glob("*.json") if d.exists() else []:
        c = _read_claim(local, p.stem)
        if c and c.get("session") == session and not c.get("released"):
            _write_claim(local, p.stem, session, now=now)


def retry_hf(fn, what, attempts=8, base_s=30.0, fatal=(), sleep=time.sleep, log=None):
    """Run ``fn()``; when Hugging Face answers with a rate limit (429 Too Many Requests) or the network hiccups, wait and try again
    (30 s, 60 s, 2 min, 4 min, 5 min ...). Downloads are resumable (finished files are skipped), so every attempt continues where the
    last one stopped. Exceptions listed in ``fatal`` (a repo that does not exist) are raised at once. After the third failure the
    Xet download path (the one that was rate limited) is switched off for the remaining attempts, which falls back to plain HTTPS."""
    log = log or (lambda m, **kw: print(m, flush=True))
    for i in range(1, attempts + 1):
        try:
            return fn()
        except fatal:
            raise
        except Exception as e:
            if i == attempts:
                raise
            wait = min(300.0, base_s * 2 ** (i - 1))
            log(f"[sync] {what}: attempt {i}/{attempts} failed ({type(e).__name__}: {str(e)[:160]}); waiting {wait:.0f} s and retrying")
            if i == 3:
                try:
                    import huggingface_hub.constants as hc
                    hc.HF_HUB_DISABLE_XET = True
                    log("[sync] switching off the Xet download path (plain HTTPS from now on)")
                except Exception:
                    pass
            sleep(wait)


def parse_spec(spec):
    kind, _, rest = spec.partition(":")
    if kind not in ("hf", "dir") or not rest:
        raise ValueError(f"sync spec must be 'hf:<user>/<repo>' or 'dir:<path>', got {spec!r}")
    return kind, rest


def _copy_atomic(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


class Syncer:
    def __init__(self, spec, local_dir, token=None, session=None):
        self.kind, self.target = parse_spec(spec)
        self.local = Path(local_dir)
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.session = session or os.environ.get("TS_SESSION_ID")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.last_error = None

    def _flock(self):
        return file_lock(self.local / LOCK)

    def _remember(self, keys):
        """Merge {rel: key} into the on-disk baseline (read-modify-write; callers hold the file lock)."""
        b = load_baseline(self.local) or {}
        b.update(keys)
        save_baseline(self.local, b)

    # -- directory backend ------------------------------------------------------------------------------------------
    def _push_dir(self):
        dst, keys = Path(self.target), {}
        for rel, f in changed_files(self.local, load_baseline(self.local)).items():
            keys[rel] = _stat_key(f)
            _copy_atomic(f, dst / rel)
        self._remember(keys)
        return len(keys)

    def _pull_dir(self, only=None):
        src = Path(self.target)
        n, keys = 0, {}
        for rel, f in local_files(src).items():
            if only is not None and not only(rel):
                continue
            d = self.local / rel
            tiny = rel.startswith((".done/", CLAIM_DIR + "/"))                  # markers: compare content, mtimes are too coarse
            newer = (d.exists() and tiny and d.read_bytes() != f.read_bytes()) or (
                d.exists() and not tiny and f.stat().st_mtime > d.stat().st_mtime + 1)
            if not d.exists() or newer:
                _copy_atomic(f, d); n += 1
            if d.exists() and (tiny or d.stat().st_mtime <= f.stat().st_mtime + 1):   # identical to the storage: not ours to push
                keys[rel] = _stat_key(d)
        self._remember(keys)
        return n

    # -- Hugging Face backend ---------------------------------------------------------------------------------------
    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def _push_hf(self, squash):
        from huggingface_hub import CommitOperationAdd
        api = self._api()
        api.create_repo(self.target, repo_type="model", private=True, exist_ok=True)
        changed = changed_files(self.local, load_baseline(self.local))
        if not changed:
            return 0
        keys = {rel: _stat_key(f) for rel, f in changed.items()}        # taken BEFORE the upload: a file that changes
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(f)) for rel, f in changed.items()]  # meanwhile is re-sent
        api.create_commit(repo_id=self.target, repo_type="model", operations=ops,
                          commit_message=f"Run state at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        self._remember(keys)
        if squash:                                    # old checkpoint versions would otherwise pile up in the repo
            try:
                api.super_squash_history(self.target, repo_type="model")
            except Exception as e:                    # not fatal: only saves space
                print(f"[sync] could not squash history: {e}", flush=True)
        return len(changed)

    def _pull_hf(self, only=None):
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
        self.local.mkdir(parents=True, exist_ok=True)
        patterns = None if only is None else [CLAIM_DIR + "/*", ".done/*"]
        try:
            retry_hf(lambda: snapshot_download(self.target, repo_type="model", local_dir=str(self.local), token=self.token,
                                               allow_patterns=patterns, max_workers=4),
                     "restore of the run state", fatal=(RepositoryNotFoundError,))
            remote = set(self._api().list_repo_files(self.target, repo_type="model"))
        except RepositoryNotFoundError:
            print(f"[sync] repo {self.target} does not exist yet: starting from scratch", flush=True)
            return 0
        mine = {rel: f for rel, f in local_files(self.local).items()
                if only is None or rel.startswith((".done/", CLAIM_DIR + "/"))}   # a claims-only pull must not hide newer local files
        self._remember({rel: _stat_key(f) for rel, f in mine.items() if rel in remote})
        return len(mine)

    # -- public -----------------------------------------------------------------------------------------------------
    def push(self, squash=False):
        with self._lock, self._flock():
            if self.session:
                heartbeat(self.local, self.session)
            n = self._push_dir() if self.kind == "dir" else self._push_hf(squash)
        print(f"[sync] pushed ({n} files) to {self.kind}:{self.target}", flush=True)
        return n

    def pull(self, claims_only=False):
        """Fetch the run state. ``claims_only`` fetches just the stage markers and claims (cheap, used before a stage starts)."""
        only = (lambda rel: rel.startswith((".done/", CLAIM_DIR + "/"))) if claims_only else None
        with self._lock, self._flock():
            n = self._pull_dir(only) if self.kind == "dir" else self._pull_hf(only)
        print(f"[sync] pulled ({n} files) from {self.kind}:{self.target}", flush=True)
        return n

    def try_claim(self, stage, ttl_min=CLAIM_TTL_MIN):
        """Refresh the view of the storage, then take the claim on ``stage`` and publish it. False = skip this stage."""
        if not self.session:
            raise ValueError("claiming needs a session id (TS_SESSION_ID)")
        try:
            self.pull(claims_only=True)
        except Exception as e:                        # offline: another session cannot be seen; carry on and claim locally
            print(f"[sync] could not refresh claims ({type(e).__name__}: {e}); claiming locally", flush=True)
        if not claim_stage(self.local, stage, self.session, ttl_min):
            return False
        try:
            self.push()
        except Exception as e:
            print(f"[sync] claim not published yet ({type(e).__name__}: {e}); the next push will", flush=True)
        return True

    def start_background(self, interval_min=30.0):
        """Push every ``interval_min`` minutes on a daemon thread. A failed push is remembered, not fatal."""
        def loop():
            while not self._stop.wait(interval_min * 60):
                try:
                    self.push()
                    self.last_error = None
                except Exception as e:
                    self.last_error = e
                    print(f"[sync] background push failed (will retry): {e}", flush=True)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop_background(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


DATA_PATTERNS = ["raw/**", "processed/*.txt", "processed/*.csv"]


def raw_present(data):
    """Are the raw training images in <data>/raw? (the split lists data/processed/*.txt come with the git clone, so they prove nothing)"""
    raw = Path(data) / "raw"
    if not raw.is_dir():
        return False
    for d in raw.iterdir():
        if d.is_dir() and any(f.is_file() and f.name != ".gitkeep" for f in d.rglob("*")):
            return True
    return False


def _hf_token(token=None):
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def data_push(repo, local_data, token=None):
    """Upload data/raw and the split lists once to a private Hugging Face *dataset* repo (resumable, any size up to the
    account's storage). A later session restores them with ``data_pull`` in minutes instead of re-downloading 23 GB from the
    original sources."""
    from huggingface_hub import HfApi
    api = HfApi(token=_hf_token(token))
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(local_data), allow_patterns=DATA_PATTERNS)
    print(f"[sync] data uploaded to dataset {repo}", flush=True)


def data_pull(repo, local_data, token=None):
    """Download the data uploaded by ``data_push`` into ``local_data`` (skips files that are already complete)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError
    Path(local_data).mkdir(parents=True, exist_ok=True)
    try:                                                              # 6,000+ files: a few at a time, and retry on a rate limit (HTTP 429)
        retry_hf(lambda: snapshot_download(repo, repo_type="dataset", local_dir=str(local_data), token=_hf_token(token),
                                           allow_patterns=DATA_PATTERNS, max_workers=4),
                 "restore of the dataset", fatal=(RepositoryNotFoundError,))
    except RepositoryNotFoundError:
        print(f"[sync] dataset {repo} does not exist yet", flush=True)
        return False
    if not raw_present(local_data):                                  # a missing repo does not always raise: the local folder is returned
        print(f"[sync] dataset {repo} holds no images (it does not exist yet, or was never uploaded)", flush=True)
        return False
    print(f"[sync] data restored from dataset {repo}", flush=True)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pull", "push", "data-pull", "data-push", "claim", "release"])
    ap.add_argument("--data-repo", default=os.environ.get("TS_DATA_REPO"), help="Hugging Face dataset repo <user>/<name> (data-pull / data-push)")
    ap.add_argument("--data-dir", default="data", help="local data folder (contains raw/ and processed/)")
    ap.add_argument("--spec", default=os.environ.get("TS_SYNC"), help="hf:<user>/<repo> or dir:<path> (default: $TS_SYNC)")
    ap.add_argument("--dir", default="experiments", help="local run folder")
    ap.add_argument("--stage", default=None, help="claim / release: the pipeline stage")
    ap.add_argument("--squash", action="store_true", help="hf push: drop old versions of the files afterwards")
    a = ap.parse_args(argv)
    if a.action in ("data-pull", "data-push"):
        if not a.data_repo:
            sys.exit("no dataset repo given: pass --data-repo <user>/<name> or set TS_DATA_REPO")
        (data_pull if a.action == "data-pull" else data_push)(a.data_repo, a.data_dir)
        return
    if not a.spec:
        sys.exit("no storage given: pass --spec or set TS_SYNC (for example hf:<user>/<repo>)")
    s = Syncer(a.spec, a.dir, session=new_session_id())
    if a.action == "claim":                           # exit 0 = this session owns the stage now, 3 = skip it (done / owned by another)
        if not a.stage:
            sys.exit("claim needs --stage")
        ok = s.try_claim(a.stage, float(os.environ.get("TS_CLAIM_TTL_MIN", CLAIM_TTL_MIN)))
        print(f"[sync] stage {a.stage}: " + ("claimed" if ok else "skipped (finished or running in another session)"), flush=True)
        sys.exit(0 if ok else 3)
    if a.action == "release":
        if a.stage:
            release_stage(s.local, a.stage, s.session)
        else:
            release_all(s.local, s.session)
        s.push()
        return
    s.pull() if a.action == "pull" else s.push(squash=a.squash)


if __name__ == "__main__":
    main()
