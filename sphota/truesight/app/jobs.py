"""Jobs of the web service: an append-only event list per job (replayed to every subscriber of the SSE stream), a folder of
files per session, and a clean-up of old sessions. Pure Python, no torch, so it is testable anywhere."""
import shutil
import threading
import time
import uuid
from pathlib import Path

TERMINAL = ("done", "error")


class Job:
    """One unit of work. ``emit`` appends an event dict (it needs a ``stage`` key); ``events_from(i)`` returns the events from
    index i on. A job ends with a ``done`` or an ``error`` event; nothing is emitted after that."""

    def __init__(self, job_id, folder, parent=None):
        self.id, self.dir, self.parent = job_id, Path(folder), parent
        self.created = time.time()
        self.events = []
        self.meta = {}                      # small facts the engine keeps about the session (sizes, mode, boxes ...)
        self._cond = threading.Condition()

    @property
    def finished(self):
        return bool(self.events) and self.events[-1]["stage"] in TERMINAL

    def emit(self, event):
        with self._cond:
            if self.finished:
                return
            self.events.append(event)
            self._cond.notify_all()

    def fail(self, message):
        self.emit({"stage": "error", "message": str(message)})

    def events_from(self, i):
        with self._cond:
            return list(self.events[i:])

    def wait(self, i, timeout):
        """Block until there is an event at index >= i (or the timeout); used by tests and thread-side waiters."""
        with self._cond:
            self._cond.wait_for(lambda: len(self.events) > i, timeout)
            return list(self.events[i:])

    def url(self, name):
        return f"/api/jobs/{self.root_id}/files/{name}"

    @property
    def root_id(self):
        return self.parent.id if self.parent is not None else self.id

    def path(self, name):
        """Path of a file of this session. The name must be a plain file name (no folders), so a request can never leave
        the session folder."""
        if not name or name != Path(name).name or name in (".", ".."):
            raise ValueError("bad file name")
        return self.dir / name


class JobStore:
    def __init__(self, root, ttl_s=6 * 3600):
        self.root, self.ttl = Path(root), ttl_s
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs = {}
        self._lock = threading.Lock()
        self._runs = {}

    def create(self):
        jid = uuid.uuid4().hex[:12]
        folder = self.root / jid
        folder.mkdir(parents=True, exist_ok=True)
        job = Job(jid, folder)
        with self._lock:
            self._jobs[jid] = job
        return job

    def create_run(self, parent):
        """A child job (one re-decode of the same session): its own event stream, the files of the parent session."""
        with self._lock:
            n = self._runs[parent.id] = self._runs.get(parent.id, 0) + 1
            job = Job(f"{parent.id}-d{n}", parent.dir, parent=parent)
            self._jobs[job.id] = job
        return job

    def get(self, jid):
        with self._lock:
            return self._jobs.get(jid)

    def sweep(self, now=None):
        """Delete sessions older than the time to live. Returns the number removed."""
        now = now or time.time()
        with self._lock:
            old = [j for j in self._jobs.values() if j.parent is None and now - j.created > self.ttl]
            for j in old:
                for k in [k for k, v in self._jobs.items() if v is j or v.parent is j]:
                    self._jobs.pop(k, None)
        for j in old:
            shutil.rmtree(j.dir, ignore_errors=True)
        return len(old)
