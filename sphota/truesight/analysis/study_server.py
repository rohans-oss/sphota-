"""A tiny web page for the user study (standard library only, no Gradio needed).

    python scripts/user_study_app.py --study study --port 8000
    # participants open http://<this machine>:8000 (same Wi-Fi), or expose the port with a tunnel such as ngrok

A participant types a short id (letters, digits, - and _), then sees their 20 pairs one by one: two decoded images side by side
(left / right assignment is random and stored, not shown), the original below, and the two questions of the guide. Every
answer is appended to <study>/responses.csv immediately, so closing the browser or restarting the server loses nothing and a
participant who comes back with the same id continues at the next unanswered pair.
"""
import html
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import pandas as pd

from .user_study import (PAIRS_PER_PERSON, Q1_CHOICES, Q2_CHOICES, append_response, assign_trials, done_trials)

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_LOCK = threading.Lock()
CSS = """body{font-family:system-ui,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;color:#1d1d1f}
.row{display:flex;gap:1rem;justify-content:center}.row figure{margin:0;text-align:center;flex:1}
img{max-width:100%;border:1px solid #ccc;background:#fff}fieldset{margin:1rem 0;border:1px solid #ccc;padding:.6rem 1rem}
label{display:inline-block;margin-right:1.2rem;font-size:1.05rem}button{font-size:1.1rem;padding:.5rem 1.4rem}
.small{color:#555;font-size:.9rem}progress{width:100%}"""


def page(body, title="Image comparison study"):
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>").encode("utf-8")


def start_page(error=""):
    err = f"<p style='color:#b00'>{html.escape(error)}</p>" if error else ""
    return page(f"<h1>Image comparison study</h1><p>You will see {PAIRS_PER_PERSON} pairs of compressed images. For each pair, "
                f"choose the one that looks better and tell us whether either image shows something different from the original. "
                f"It takes about 5 minutes. Your answers are stored without your name.</p>{err}"
                f"<form action='/study' method='get'><p><label>Your id (letters / digits, for example your initials): "
                f"<input name='p' required maxlength='40' pattern='[A-Za-z0-9_-]+'></label></p><button>Start</button></form>")


def trial_page(participant, trial, total, answered):
    p = html.escape(participant)
    radios1 = "".join(f"<label><input type='radio' name='q1' value='{c}' required> {c.capitalize()} image</label>" for c in Q1_CHOICES)
    labels2 = {"neither": "Neither", "left": "Only the left image", "right": "Only the right image", "both": "Both images"}
    radios2 = "".join(f"<label><input type='radio' name='q2' value='{c}' required> {labels2[c]}</label>" for c in Q2_CHOICES)
    img = lambda rel: f"/img/{quote(rel)}"  # noqa: E731
    return page(
        f"<progress value='{answered}' max='{total}'></progress><p class='small'>Pair {trial['trial']} of {total} - participant {p}</p>"
        f"<div class='row'><figure><img src='{img(trial['left'])}'><figcaption>Left</figcaption></figure>"
        f"<figure><img src='{img(trial['right'])}'><figcaption>Right</figcaption></figure></div>"
        f"<form action='/answer' method='post'><input type='hidden' name='p' value='{p}'>"
        f"<input type='hidden' name='trial' value='{trial['trial']}'><input type='hidden' name='t0' value='{time.time():.3f}'>"
        f"<fieldset><legend><b>1. Which image looks better?</b></legend>{radios1}</fieldset>"
        f"<fieldset><legend><b>2. Does either image show something different from the original (below)?</b> "
        f"Look for changed text, faces, shapes or objects.</legend>{radios2}</fieldset>"
        f"<button>Next</button></form><h3>Original</h3><p><img src='{img(trial['original'])}' style='max-width:55%'></p>")


class Study:
    """Study state: the pair table, the trials of each participant and the responses file."""

    def __init__(self, folder, n_pairs=PAIRS_PER_PERSON, seed=0):
        self.dir = Path(folder).resolve()
        self.pairs = pd.read_csv(self.dir / "pairs.csv")
        self.responses = self.dir / "responses.csv"
        self.n_pairs, self.seed = n_pairs, seed

    def trials(self, participant):
        return assign_trials(self.pairs, participant, self.n_pairs, self.seed)

    def next_trial(self, participant):
        done = done_trials(self.responses, participant)
        trials = self.trials(participant)
        return next((t for t in trials if t["trial"] not in done), None), len(trials), len(done)

    def record(self, participant, trial_no, q1, q2, t0):
        t = next((t for t in self.trials(participant) if t["trial"] == trial_no), None)
        if t is None:
            raise ValueError("unknown trial")
        with _LOCK:
            if trial_no in done_trials(self.responses, participant):         # a double click or a back-button resubmit
                return False
            append_response(self.responses, participant, trial_no, t["pair_id"], t["left_role"], q1, q2, max(0.0, time.time() - t0))
        return True

    def image_path(self, rel):
        p = (self.dir / unquote(rel)).resolve()
        if self.dir not in p.parents or not p.is_file():                      # no path tricks, no files outside the study folder
            return None
        return p


def make_handler(study):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8", headers=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):               # keep the console quiet
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, start_page())
            if u.path == "/study":
                p = parse_qs(u.query).get("p", [""])[0]
                if not ID_RE.match(p):
                    return self._send(400, start_page("Please use letters, digits, - or _ (at most 40 characters)."))
                trial, total, answered = study.next_trial(p)
                if trial is None:
                    return self._send(200, page(f"<h1>Thank you!</h1><p>You answered all {total} pairs.</p>"))
                return self._send(200, trial_page(p, trial, total, answered))
            if u.path.startswith("/img/"):
                f = study.image_path(u.path[len("/img/"):])
                if f is None:
                    return self._send(404, b"not found", "text/plain")
                ctype = "image/png" if f.suffix.lower() == ".png" else "image/jpeg"
                return self._send(200, f.read_bytes(), ctype, [("Cache-Control", "max-age=3600")])
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/answer":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            f = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode("utf-8")).items()}
            try:
                if not ID_RE.match(f.get("p", "")):
                    raise ValueError("bad participant id")
                study.record(f["p"], int(f["trial"]), f.get("q1", ""), f.get("q2", ""), float(f.get("t0", time.time())))
            except (ValueError, KeyError) as e:
                return self._send(400, page(f"<p>Could not save the answer: {html.escape(str(e))}</p><p><a href='/'>Back</a></p>"))
            self._send(303, b"", headers=[("Location", f"/study?p={quote(f['p'])}")])
    return Handler


def serve(folder, host="0.0.0.0", port=8000, n_pairs=PAIRS_PER_PERSON, seed=0):
    """Start the study server (blocks). ``port=0`` picks a free port; returns the server in a thread via ``serve_background``."""
    study = Study(folder, n_pairs, seed)
    srv = ThreadingHTTPServer((host, port), make_handler(study))
    print(f"study server on http://{host}:{srv.server_address[1]}  (answers go to {study.responses})", flush=True)
    srv.serve_forever()


def serve_background(folder, host="127.0.0.1", port=0, n_pairs=PAIRS_PER_PERSON, seed=0):
    """Same server on a daemon thread (used by the tests). Returns (server, base_url)."""
    srv = ThreadingHTTPServer((host, port), make_handler(Study(folder, n_pairs, seed)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://{host}:{srv.server_address[1]}"
