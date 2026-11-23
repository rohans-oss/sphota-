"""Pure helpers for tuning the guidance and checking the Step 12 done-criteria (unit-tested, no models needed)."""
import math

import numpy as np

CONSISTENCY_TARGET = 0.95
TFR_FACTOR_TARGET = 2.0          # Sphoṭa TFR >= 2x plain diffusion
FIR_FACTOR_TARGET = 1.5          # Sphoṭa FIR >= 1.5x plain diffusion
LPIPS_SLACK = 0.10               # LPIPS at most ~10% worse than the unguided decode
MIN_EXAMPLES = 6                 # visual examples where plain diffusion changed text/faces and Sphoṭa did not
METHODS = ("coarse", "plain", "roi_only", "guided_only", "sphota")


def choose_guidance_config(grid, baseline_lpips, slack=LPIPS_SLACK):
    """Freeze (lr, t_start) from a tuning grid.

    ``grid`` is a DataFrame with columns lr, t_start, consistency, lpips (and optionally psnr). Rule: among configurations
    whose mean LPIPS is within ``slack`` of the unguided baseline, take the highest consistency rate (ties: smaller lr,
    then smaller t_start - the gentler setting). If none is within the slack, take the lowest LPIPS. Too strong guidance
    makes the output look like the blurry coarse decode; too weak brings hallucinations back, hence the LPIPS limit.
    Returns (config dict, reason string)."""
    limit = baseline_lpips * (1 + slack)
    ok = grid[grid.lpips <= limit]
    if ok.empty:
        best = grid.sort_values(["lpips", "lr", "t_start"]).iloc[0]
        why = (f"no configuration keeps LPIPS within {slack:.0%} of the unguided {baseline_lpips:.4f} "
               f"(limit {limit:.4f}); took the lowest LPIPS")
    else:
        best = ok.sort_values(["consistency", "lr", "t_start"], ascending=[False, True, True]).iloc[0]
        why = (f"highest consistency ({best.consistency:.3f}) among {len(ok)}/{len(grid)} configurations with "
               f"LPIPS <= {limit:.4f} ({slack:.0%} above the unguided {baseline_lpips:.4f})")
    return {"lr": float(best.lr), "t_start": int(best.t_start), "consistency": float(best.consistency),
            "lpips": float(best.lpips)}, why


def safe_ratio(a, b):
    """a / b; inf when b == 0 and a > 0; NaN when both are 0 or either is NaN."""
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return float("nan")
    if b == 0:
        return float("inf") if a > 0 else float("nan")
    return a / b


def lpips_degradation_pct(guided, unguided):
    """Relative LPIPS change in % (positive = worse)."""
    return 100.0 * (guided / unguided - 1.0)


def wins_by_pass(plain_passed, sphota_passed):
    """Indices where plain diffusion failed and Sphoṭa passed (lists of bool or None for 'not applicable')."""
    return [i for i, (p, s) in enumerate(zip(plain_passed, sphota_passed)) if p is False and s is True]


def wins_by_cosine(plain_cos, sphota_cos, threshold=0.7):
    """Indices where plain diffusion changed the face (cosine < threshold) and Sphoṭa kept it (>= threshold)."""
    return [i for i, (p, s) in enumerate(zip(plain_cos, sphota_cos)) if p < threshold <= s]


def done_criteria(table, n_examples=None):
    """Check the Step 12 targets. ``table`` maps a method name to a dict with consistency, tfr, fir, lpips (any may be
    missing / NaN). Compared pairs: Sphoṭa vs plain (TFR, FIR, consistency) and Sphoṭa vs the same boxes without guidance
    ('roi_only', ablation A4) for the LPIPS cost of the guidance."""
    s, p = table.get("sphota", {}), table.get("plain", {})
    out = {}
    if "consistency" in s:
        out["consistency"] = {"value": s["consistency"], "target": CONSISTENCY_TARGET, "met": bool(s["consistency"] >= CONSISTENCY_TARGET)}
        if "consistency" in p:
            out["consistency"]["plain"] = p["consistency"]
    for key, target in (("tfr", TFR_FACTOR_TARGET), ("fir", FIR_FACTOR_TARGET)):
        if key in s and key in p:
            r = safe_ratio(s[key], p[key])
            out[key + "_factor"] = {"sphota": s[key], "plain": p[key], "factor": r, "target": target,
                                    "met": bool(not math.isnan(r) and r >= target)}
    base = table.get("roi_only") or table.get("plain")
    if base and "lpips" in base and "lpips" in s:
        d = lpips_degradation_pct(s["lpips"], base["lpips"])
        out["lpips_cost_pct"] = {"value": d, "limit": LPIPS_SLACK * 100, "met": bool(d <= LPIPS_SLACK * 100),
                                 "reference": "roi_only" if table.get("roi_only") else "plain"}
    if n_examples is not None:
        out["visual_examples"] = {"value": n_examples, "target": MIN_EXAMPLES, "met": bool(n_examples >= MIN_EXAMPLES)}
    return out


def mean_history(histories):
    """Average guided-step histories of several images (each a list of (t, lb_before, lro_before, lb_after, lro_after)
    with identical t sequences). Returns an array (n_steps, 5)."""
    hs = [np.asarray(h, dtype=float) for h in histories if len(h)]
    if not hs:
        return np.zeros((0, 5))
    n = min(len(h) for h in hs)
    return np.mean([h[:n] for h in hs], axis=0)
