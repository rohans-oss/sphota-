"""Pure helpers for the Step 10 evaluation (kept out of the script so they can be unit-tested)."""
import math

import numpy as np
import pandas as pd

LPIPS_TARGET_PCT = 25.0          # diffusion decode must have >= 25% lower LPIPS than the coarse decode at the same bpp


def improvement_pct(coarse, diffusion):
    """Relative reduction in % of a lower-is-better metric: 100 * (1 - diffusion / coarse)."""
    return 100.0 * (1.0 - diffusion / coarse)


def choose_default_w(sweep, lo=1.0, hi=2.0):
    """Pick the default CFG scale from a sweep DataFrame with columns w, lpips, psnr.

    Rule: among w in [lo, hi] (the usable range given in the guide: larger w is sharper but riskier for faithfulness)
    take the lowest mean LPIPS; ties go to the smaller w. Returns (w, reason). The reason also reports the PSNR cost
    relative to w = 1 so the trade-off is written down; confirm it on the sample grid."""
    cand = sweep[(sweep.w >= lo) & (sweep.w <= hi)].sort_values(["lpips", "w"])
    if cand.empty:
        raise ValueError(f"no w in [{lo}, {hi}] in the sweep")
    best = cand.iloc[0]
    ref = sweep[np.isclose(sweep.w, 1.0)]
    note = ""
    if len(ref):
        r = ref.iloc[0]
        note = f" (vs w=1: LPIPS {best.lpips - r.lpips:+.4f}, PSNR {best.psnr - r.psnr:+.2f} dB)"
    return float(best.w), (f"w={best.w:g} has the lowest mean LPIPS ({best.lpips:.4f}) among w in [{lo:g}, {hi:g}]"
                           f"{note}; larger w is sharper but moves the decode further from the transmitted content.")


def summarize(per_image, methods=("coarse", "diffusion", "jpeg", "webp")):
    """Mean of each metric per method from a long DataFrame (columns: method, bpp, psnr, msssim, lpips, ...)."""
    cols = [c for c in ("bpp", "psnr", "msssim", "lpips", "time") if c in per_image]
    out = per_image[per_image.method.isin(methods)].groupby("method")[cols].mean().reindex(
        [m for m in methods if m in set(per_image.method)])
    return out


def checks(summary, fid=None):
    """The Step 10 done-criteria that can be read off the summary (and optional FID dict {method: value})."""
    res = {}
    if {"coarse", "diffusion"} <= set(summary.index):
        imp = improvement_pct(summary.loc["coarse", "lpips"], summary.loc["diffusion", "lpips"])
        res["lpips_reduction_pct"] = float(imp)
        res["lpips_target_met"] = bool(imp >= LPIPS_TARGET_PCT)
    if fid and "diffusion" in fid:
        others = [m for m in ("jpeg", "webp", "coarse") if m in fid]
        res["fid_beats_jpeg_webp_coarse"] = bool(others) and all(fid["diffusion"] < fid[m] for m in others)
    return res


def fmt_table(df, floatfmt="{:.4f}"):
    """GitHub-markdown table of a small DataFrame."""
    cols = [df.index.name or "method"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not math.isnan(v)
                                                      else str(v) for v in r.values]) + " |")
    return "\n".join(lines)
