"""User study: pair assignment, statistics and the web page (truesight/analysis/user_study.py, study_server.py)."""
import hashlib
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from truesight.analysis import user_study as U
from truesight.analysis.study_server import serve_background


def _pairs(n_images=12, baselines=("jpeg", "webp", "plain")):
    rows = []
    for i in range(n_images):
        for b in baselines:
            pid = f"{i:03d}_{b}"
            nm = lambda role: "images/" + hashlib.sha256(f"{pid}|{role}".encode()).hexdigest()[:14] + ".png"  # noqa: E731
            rows.append(dict(pair_id=pid, image=f"img{i}.png", baseline=b, original=f"images/o{i}.png", sphota=nm("s"), other=nm("x"),
                             bpp_sphota=0.04, bpp_other=0.06))
    return pd.DataFrame(rows)


def test_binom_ci_known_values():
    lo, hi = U.binom_ci(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-3) and hi == pytest.approx(0.8129, abs=1e-3)
    assert U.binom_ci(0, 10)[0] == 0.0 and U.binom_ci(0, 10)[1] == pytest.approx(0.3085, abs=1e-3)
    assert U.binom_ci(10, 10)[1] == 1.0 and np.isnan(U.binom_ci(0, 0)[0])
    assert U.sign_test_p(5, 10) == pytest.approx(1.0) and U.sign_test_p(20, 20) < 1e-5


def test_assign_trials_is_deterministic_balanced_and_blind():
    pairs = _pairs(25)
    t1, t2 = U.assign_trials(pairs, "alice"), U.assign_trials(pairs, "alice")
    assert t1 == t2 and len(t1) == 20 and [t["trial"] for t in t1] == list(range(1, 21))
    assert len({t["pair_id"] for t in t1}) == 20
    assert len({pairs.set_index("pair_id").loc[t["pair_id"], "image"] for t in t1}) == 20      # no image repeats for one person
    base = Counter(t["pair_id"].split("_")[1] for t in t1)
    assert set(base) == {"jpeg", "webp", "plain"} and max(base.values()) - min(base.values()) <= 1
    assert U.assign_trials(pairs, "bob") != t1
    left_sphota = [t["left_role"] == "sphota" for p in range(60) for t in U.assign_trials(pairs, f"p{p}")]
    assert 0.4 < np.mean(left_sphota) < 0.6                                                       # random left / right
    for t in t1:                                                                                  # left / right really follow left_role
        row = pairs.set_index("pair_id").loc[t["pair_id"]]
        assert t["left"] == (row.sphota if t["left_role"] == "sphota" else row.other)
        assert t["right"] == (row.other if t["left_role"] == "sphota" else row.sphota)


def _answers(pairs, n_people=10, prefer=1.0, notice_sphota=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_people):
        for t in U.assign_trials(pairs, f"p{p}"):
            sph_side = "left" if t["left_role"] == "sphota" else "right"
            other_side = "right" if sph_side == "left" else "left"
            wants = rng.random() < prefer
            rows.append(dict(participant=f"p{p}", trial=t["trial"], pair_id=t["pair_id"], left_role=t["left_role"],
                             q1_choice=sph_side if wants else other_side,
                             q2_changed=sph_side if rng.random() < notice_sphota else "neither", seconds=3.0, timestamp="t"))
    return pd.DataFrame(rows)


def test_analysis_maps_left_right_to_sphota_and_reports_intervals():
    pairs = _pairs()
    table, design = U.analyse(_answers(pairs, 10, prefer=1.0), pairs)
    allr = table[table.group == "all"].iloc[0]
    assert allr.pairs == 200 and allr.preference_pct == 100.0 and allr.preference_ci_low > 98 and allr.sign_test_p < 1e-30
    assert allr.notice_any_pct == 0.0 and set(table.group) == {"all", "jpeg", "webp", "plain"}
    assert design["meets_design"] and design["participants"] == 10 and design["pairs_per_participant_min"] == 20
    t2, _ = U.analyse(_answers(pairs, 10, prefer=0.0, notice_sphota=1.0), pairs)
    r = t2[t2.group == "all"].iloc[0]
    assert r.preference_pct == 0.0 and r.notice_sphota_pct == 100.0 and r.notice_other_pct == 0.0 and r.notice_any_pct == 100.0
    md = U.fmt_markdown(table, design)
    assert "meets" in md and "95% CI" in md


def test_design_check_flags_too_few_people_or_pairs():
    pairs = _pairs()
    _, d = U.analyse(_answers(pairs, 4), pairs)
    assert not d["meets_design"] and not d["meets_min_participants"] and d["meets_pairs_per_person"]
    short = _answers(pairs, 10)
    short = short[short.trial <= 15]
    _, d2 = U.analyse(short, pairs)
    assert d2["meets_min_participants"] and not d2["meets_pairs_per_person"]


def test_append_response_validates(tmp_path):
    f = tmp_path / "r.csv"
    U.append_response(f, "a", 1, "000_jpeg", "sphota", "left", "neither", 2.5)
    for bad in (("a", 2, "x", "sphota", "up", "neither"), ("a", 2, "x", "sphota", "left", "maybe"), ("a", 2, "x", "middle", "left", "both")):
        with pytest.raises(ValueError):
            U.append_response(f, *bad, 1.0)
    assert len(pd.read_csv(f)) == 1 and U.done_trials(f, "a") == {1} and U.done_trials(f, "b") == set()


# ------------------------------------------------------------------------------------------------------- the web page
def _study_dir(tmp_path, n_images=12):
    d = tmp_path / "study"
    (d / "images").mkdir(parents=True)
    pairs = _pairs(n_images)
    rng = np.random.default_rng(0)
    paths = set(pairs.original) | set(pairs.sphota) | set(pairs.other)
    for rel in paths:
        Image.fromarray((rng.random((8, 8, 3)) * 255).astype(np.uint8)).save(d / rel)
    pairs.to_csv(d / "pairs.csv", index=False)
    (tmp_path / "secret.txt").write_text("do not serve")
    return d


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url, data):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    op = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, urllib.parse.urlencode(data).encode(), method="POST")
    try:
        r = op.open(req, timeout=10)
        return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)


def test_web_page_walks_a_participant_through_20_pairs_and_resumes(tmp_path):
    d = _study_dir(tmp_path)
    srv, base = serve_background(d)
    try:
        assert b"Image comparison study" in _get(base + "/")[1]
        answered = 0
        for i in range(20):
            status, body = _get(f"{base}/study?p=carol")
            page = body.decode()
            assert status == 200 and f"Pair {i + 1} of 20" in page
            assert "sphota" not in page.lower().replace("image comparison", "") and "jpeg" not in page.lower()       # nothing reveals the method
            trial = int(re.search(r"name='trial' value='(\d+)'", page).group(1))
            t0 = re.search(r"name='t0' value='([\d.]+)'", page).group(1)
            for src in re.findall(r"<img src='([^']+)'", page):
                assert _get(base + src)[0] == 200
            code, hdr = _post(base + "/answer", dict(p="carol", trial=trial, t0=t0, q1="left", q2="neither"))
            assert code == 303 and "/study?p=carol" in hdr["Location"]
            answered += 1
            if i == 4:                                                                                             # "closed the browser"
                code, _ = _post(base + "/answer", dict(p="carol", trial=trial, t0=t0, q1="right", q2="both"))       # double submit
                assert code == 303
        assert b"Thank you" in _get(f"{base}/study?p=carol")[1]
        r = pd.read_csv(d / "responses.csv")
        assert len(r) == 20 and r.trial.tolist() == list(range(1, 21)) and r.participant.eq("carol").all()
        assert r.q1_choice.eq("left").all()                                                                         # the duplicate was ignored
        pairs = pd.read_csv(d / "pairs.csv")
        table, _ = U.analyse(r, pairs)
        assert table.iloc[0].pairs == 20

        _post(base + "/answer", dict(p="dave", trial=1, t0="1", q1="left", q2="neither"))
        status, body = _get(f"{base}/study?p=dave")
        assert "Pair 2 of 20" in body.decode()                                                                      # resumes at the next pair
        assert _post(base + "/answer", dict(p="dave", trial=2, t0="1", q1="up", q2="neither"))[0] == 400            # invalid answer
        assert _post(base + "/answer", dict(p="../x", trial=2, t0="1", q1="left", q2="both"))[0] == 400            # invalid id
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/study?p=bad%20id")
        assert e.value.code == 400
    finally:
        srv.shutdown()


def test_web_page_does_not_serve_files_outside_the_study_folder(tmp_path):
    d = _study_dir(tmp_path)
    srv, base = serve_background(d)
    try:
        for bad in ("/img/../secret.txt", "/img/..%2Fsecret.txt", "/img/%2e%2e/secret.txt", "/img/pairs.csv.nope"):
            try:
                code, body = _get(base + bad)
            except urllib.error.HTTPError as e:
                code, body = e.code, b""
            assert code == 404 and b"do not serve" not in body
        assert _get(base + "/img/images/o0.png")[0] == 200
    finally:
        srv.shutdown()
