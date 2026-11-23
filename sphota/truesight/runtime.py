"""Run-time safety for long training on time-limited or unreliable machines (Kaggle sessions stop after 12 hours).

Three guarantees, used by every trainer in this project:

1. Time budget. ``TS_TIME_LIMIT_HOURS`` (set by scripts/kaggle_run.py, default 11.5) is counted from ``TS_SESSION_START``.
   When it is used up, or a SIGTERM / Ctrl-C arrives, the trainer saves its exact state and raises ``Paused``; the command
   line tools turn that into exit status 75 (``EXIT_PAUSED``) so a wrapper knows "not finished, run me again".
2. Crash safety. ``crash_safe`` wraps a training loop: if anything raises (a bug, CUDA out of memory, a NaN loss, Ctrl-C) the
   current weights, optimiser state and iteration are written to last.pt *before* the exception propagates. Fix the bug,
   run the same command, and training continues from that iteration. If the weights are already non-finite the earlier
   checkpoint is kept instead of overwriting it with garbage.
3. Time-based checkpoints. ``save_due`` fires every ``TS_SAVE_EVERY_MIN`` minutes (default 20) in addition to the
   iteration-based checkpoints, so a hard kill (the machine disappearing) loses at most that many minutes.

Environment: TS_SESSION_START (epoch seconds; default = this process's start), TS_TIME_LIMIT_HOURS (default: no limit),
TS_SAVE_EVERY_MIN (default 20).

    python -m truesight.runtime remaining          # minutes left in the budget
    python -m truesight.runtime need 30            # exit status 75 when fewer than 30 minutes are left
"""
import math
import os
import signal
import sys
import time
from contextlib import contextmanager

EXIT_PAUSED = 75                                 # EX_TEMPFAIL: "not finished, run again"


class Paused(Exception):
    """A trainer saved its state and stopped because the time budget is used up or a stop signal arrived."""


def _env_float(name, default):
    v = os.environ.get(name, "")
    try:
        return float(v) if v.strip() else default
    except ValueError:
        return default


class RunGuard:
    def __init__(self, time_limit_h=None, save_every_min=None, start=None, install_signals=True):
        self.start = start if start is not None else _env_float("TS_SESSION_START", time.time())
        self.time_limit_h = time_limit_h if time_limit_h is not None else _env_float("TS_TIME_LIMIT_HOURS", None)
        self.save_every_s = 60 * (save_every_min if save_every_min is not None else _env_float("TS_SAVE_EVERY_MIN", 20.0))
        self._signalled = False
        self._last_save = time.time()
        if install_signals:
            self._install()

    # -- signals ----------------------------------------------------------------------------------------------------
    def _install(self):
        def handler(signum, frame):
            if self._signalled:                      # a second signal means "really stop now"
                raise KeyboardInterrupt
            self._signalled = True
            print(f"\n[runtime] signal {signum}: saving and stopping at the next step", flush=True)
        for name in ("SIGTERM", "SIGINT"):
            try:
                signal.signal(getattr(signal, name), handler)
            except (ValueError, AttributeError, OSError):   # not the main thread / not supported here
                pass

    # -- budget -----------------------------------------------------------------------------------------------------
    def elapsed_min(self):
        return (time.time() - self.start) / 60.0

    def remaining_min(self):
        """Minutes left in the budget (``inf`` when there is no limit)."""
        if self.time_limit_h is None:
            return math.inf
        return self.time_limit_h * 60.0 - self.elapsed_min()

    def stop_requested(self):
        return self._signalled or self.remaining_min() <= 0

    def save_due(self):
        return time.time() - self._last_save >= self.save_every_s

    def saved(self):
        self._last_save = time.time()

    def reason(self):
        return "stop signal" if self._signalled else f"time budget of {self.time_limit_h} h used up"


_guard = None


def guard():
    global _guard
    if _guard is None:
        _guard = RunGuard()
    return _guard


def set_guard(g):
    """Replace the process-wide guard (used by tests)."""
    global _guard
    _guard = g
    return g


def finite_params(*modules):
    """True when every parameter of every module is finite (cheap sanity check before an emergency save)."""
    import torch
    for m in modules:
        for p in m.parameters():
            if not bool(torch.isfinite(p).all()):
                return False
    return True


@contextmanager
def crash_safe(save, label="train", state_ok=None):
    """Run a training loop; on any exception write the current state with ``save()`` first, then re-raise.
    ``save`` must write last.pt atomically from the loop's *current* variables. ``Paused`` is not a crash."""
    try:
        yield
    except Paused:
        raise
    except BaseException as e:                                       # includes KeyboardInterrupt and CUDA errors
        try:
            if state_ok is None or state_ok():
                save()
                print(f"[{label}] {type(e).__name__}: {e}\n[{label}] state saved to last.pt; fix the cause and run the same "
                      f"command again to continue from there", flush=True)
            else:
                print(f"[{label}] {type(e).__name__}: {e}\n[{label}] weights are not finite: keeping the previous "
                      f"checkpoint", flush=True)
        except Exception as e2:                                       # never hide the original error
            print(f"[{label}] emergency save failed: {e2}", flush=True)
        raise


def pause_if_needed(save, label="train"):
    """Call once per step: save and raise ``Paused`` when the budget is used up or a stop signal arrived."""
    g = guard()
    if g.stop_requested():
        save()
        raise Paused(f"[{label}] {g.reason()}: state saved, run again to continue")


def run_cli(main):
    """Run a script's main(); a ``Paused`` becomes exit status 75 instead of a traceback."""
    try:
        main()
    except Paused as e:
        print(f"PAUSED: {e}", flush=True)
        sys.exit(EXIT_PAUSED)


def _cli(argv):
    g = RunGuard(install_signals=False)
    if not argv or argv[0] == "remaining":
        r = g.remaining_min()
        print("inf" if math.isinf(r) else f"{r:.1f}")
        return 0
    if argv[0] == "need":
        need = float(argv[1])
        r = g.remaining_min()
        if r < need:
            print(f"only {r:.0f} min left, need {need:.0f}: pausing before this stage", flush=True)
            return EXIT_PAUSED
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
