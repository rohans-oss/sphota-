"""The small user study of section 8.5: Sphoṭa against a baseline at the same file size.

Design (from the guide): at least 10 people, 20 pairs each, random left / right order, two questions per pair:
    Q1  "Which looks better?"                                          -> forced choice, left or right
    Q2  "Does either image show something different from the original (shown below)?"  -> neither / left / right / both
Reported: the preference for Sphoṭa in %, and the share of pairs in which a change was noticed (in either image, in the
Sphoṭa image and in the baseline image), each with a 95% binomial confidence interval (Clopper-Pearson), plus a two-sided
sign test of Q1 against 50%.

Files (all under one study folder):
    pairs.csv      one row per pair: pair_id, image, baseline, original, sphota, other, bpp_sphota, bpp_other
                   (written by scripts/make_study_pairs.py; the three image paths are relative to the study folder)
    responses.csv  one row per answer: participant, trial, pair_id, left_role, q1_choice, q2_changed, seconds, timestamp
                   (written by scripts/user_study_app.py, one row at a time, so a closed browser loses nothing)
"""
import csv
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_COLUMNS = ["pair_id", "image", "baseline", "original", "sphota", "other", "bpp_sphota", "bpp_other"]
RESPONSE_COLUMNS = ["participant", "trial", "pair_id", "left_role", "q1_choice", "q2_changed", "seconds", "timestamp"]
Q1_CHOICES = ("left", "right")
Q2_CHOICES = ("neither", "left", "right", "both")
MIN_PARTICIPANTS, PAIRS_PER_PERSON = 10, 20
ROLES = ("sphota", "baseline")


def binom_ci(k, n, confidence=0.95):
    """Exact (Clopper-Pearson) confidence interval for a proportion k / n, as fractions in [0, 1]. (nan, nan) when n = 0."""
    if n <= 0:
        return float("nan"), float("nan")
    from scipy.stats import beta
    a = 1 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def sign_test_p(k, n):
    """Two-sided exact p-value of 'the preference is 50%'."""
    if n <= 0:
        return float("nan")
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5).pvalue)


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def assign_trials(pairs, participant, n_pairs=PAIRS_PER_PERSON, seed=0):
    """The 20 pairs one participant sees, in random order, each with a random left / right assignment.

    Deterministic in (participant, seed), so a person who reloads the page gets the same trials. Pairs are drawn round-robin
    over the baselines so every baseline is represented, and never repeat an image within a person while unseen images remain.
    Returns a list of dicts: trial (1-based), pair_id, left_role ('sphota' | 'baseline'), left, right, original (paths)."""
    rng = random.Random(_seed(participant, seed))
    df = pairs.reset_index(drop=True)
    by_base = {b: g.sample(frac=1.0, random_state=rng.randint(0, 2 ** 31 - 1)).to_dict("records") for b, g in df.groupby("baseline")}
    chosen, used_images, bases = [], set(), sorted(by_base)
    while len(chosen) < n_pairs and any(by_base.values()):
        for b in bases:
            pool = by_base[b]
            if not pool or len(chosen) >= n_pairs:
                continue
            idx = next((i for i, r in enumerate(pool) if r["image"] not in used_images), 0)
            r = pool.pop(idx)
            used_images.add(r["image"]); chosen.append(r)
    rng.shuffle(chosen)
    trials = []
    for i, r in enumerate(chosen, 1):
        sphota_left = rng.random() < 0.5
        trials.append(dict(trial=i, pair_id=r["pair_id"], left_role="sphota" if sphota_left else "baseline",
                           left=r["sphota"] if sphota_left else r["other"], right=r["other"] if sphota_left else r["sphota"],
                           original=r["original"]))
    return trials


def append_response(path, participant, trial, pair_id, left_role, q1, q2, seconds):
    """Append one answer (creates the file with a header). Validates the answers so a bad click cannot corrupt the data."""
    if q1 not in Q1_CHOICES or q2 not in Q2_CHOICES or left_role not in ROLES:
        raise ValueError(f"invalid answer: q1={q1!r} q2={q2!r} left_role={left_role!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RESPONSE_COLUMNS)
        w.writerow([participant, trial, pair_id, left_role, q1, q2, f"{seconds:.2f}", time.strftime("%Y-%m-%dT%H:%M:%S")])
        f.flush()


def done_trials(path, participant):
    """Trial numbers a participant already answered (to resume after a reload)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    df = pd.read_csv(p)
    return set(df.loc[df.participant.astype(str) == str(participant), "trial"].astype(int))


def prepare_responses(responses, pairs):
    """Join answers with the pair table and derive: ``prefers_sphota`` (Q1) and the Q2 flags ``noticed_any``,
    ``noticed_sphota``, ``noticed_other`` (which side the 'something different' answer points at, mapped through left_role)."""
    r = responses.copy()
    r["participant"] = r["participant"].astype(str)
    left_is_sphota = r["left_role"] == "sphota"
    chose_left = r["q1_choice"] == "left"
    r["prefers_sphota"] = (chose_left == left_is_sphota).astype(int)
    q2 = r["q2_changed"]
    r["noticed_any"] = (q2 != "neither").astype(int)
    sph_side = np.where(left_is_sphota, "left", "right")
    r["noticed_sphota"] = ((q2 == "both") | (q2 == sph_side)).astype(int)
    r["noticed_other"] = ((q2 == "both") | ((q2 != "neither") & (q2 != sph_side))).astype(int)
    return r.merge(pairs[["pair_id", "baseline", "image"]], on="pair_id", how="left")


def _rate_row(name, sub):
    n = len(sub)
    k = int(sub.prefers_sphota.sum())
    lo, hi = binom_ci(k, n)
    out = dict(group=name, pairs=n, participants=int(sub.participant.nunique()), prefers_sphota=k, preference_pct=100 * k / n if n else float("nan"),
               preference_ci_low=100 * lo, preference_ci_high=100 * hi, sign_test_p=sign_test_p(k, n))
    for col, label in (("noticed_any", "notice_any"), ("noticed_sphota", "notice_sphota"), ("noticed_other", "notice_other")):
        kk = int(sub[col].sum()); l, h = binom_ci(kk, n)
        out.update({f"{label}_pct": 100 * kk / n if n else float("nan"), f"{label}_ci_low": 100 * l, f"{label}_ci_high": 100 * h})
    return out


def analyse(responses, pairs, min_participants=MIN_PARTICIPANTS, pairs_per_person=PAIRS_PER_PERSON):
    """Table (one row for all answers and one per baseline) + the design check of the guide (>= 10 people, 20 pairs each).
    Returns (DataFrame, dict with the design check)."""
    r = prepare_responses(responses, pairs)
    rows = [_rate_row("all", r)] + [_rate_row(b, g) for b, g in r.groupby("baseline")]
    per_person = r.groupby("participant").size()
    design = {"participants": int(len(per_person)), "pairs_per_participant_min": int(per_person.min()) if len(per_person) else 0,
              "meets_min_participants": bool(len(per_person) >= min_participants),
              "meets_pairs_per_person": bool(len(per_person) and per_person.min() >= pairs_per_person),
              "answers": int(len(r))}
    design["meets_design"] = design["meets_min_participants"] and design["meets_pairs_per_person"]
    return pd.DataFrame(rows), design


def fmt_markdown(table, design):
    lines = ["| group | pairs | people | prefers Sphoṭa (95% CI) | sign test p | notices a change: any | in Sphoṭa | in baseline |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r.group} | {r.pairs} | {r.participants} | {r.preference_pct:.1f}% ({r.preference_ci_low:.1f}-{r.preference_ci_high:.1f}) | "
                     f"{r.sign_test_p:.3g} | {r.notice_any_pct:.1f}% ({r.notice_any_ci_low:.1f}-{r.notice_any_ci_high:.1f}) | "
                     f"{r.notice_sphota_pct:.1f}% | {r.notice_other_pct:.1f}% |")
    ok = "meets" if design["meets_design"] else "does NOT meet"
    lines.append(f"\nDesign: {design['participants']} participants, at least {design['pairs_per_participant_min']} pairs each: {ok} "
                 f"the guide's design (>= {MIN_PARTICIPANTS} people x {PAIRS_PER_PERSON} pairs).")
    return "\n".join(lines)
