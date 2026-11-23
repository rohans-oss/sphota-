"""Collect every measured result into the report tables (section 3.4 targets, section 8.4 ablations, faithfulness).

Everything here only READS the files the evaluation scripts already wrote under ``experiments/`` and turns them into three
things: the summary table of the guide's success targets (measured value, target, met or not), the status of the eight
ablations A1-A8, and the faithfulness table (TFR / FIR / AUROC with confidence intervals). A result that has not been measured
yet is reported as "not measured", never as a pass or a fail. Nothing is recomputed, so every number in the report can be
regenerated from the evaluation outputs of one frozen commit by ``scripts/make_figures.py``.
"""
import json
import math
from pathlib import Path

import pandas as pd

from .user_study import binom_ci

NOT_MEASURED = "not measured"
TARGET_COLUMNS = ["target", "requirement", "measured", "met", "source", "note"]

# where each evaluation script writes (relative to the experiments folder)
PATHS = dict(
    codec="step7_codec", bitstream="step8_bitstream", diffusion="step10_diffusion/eval", clic_crops="step10_diffusion/clic_crops",
    ablation="step10_ablation",
    protect="step11_protect", guidance="step12_guidance", honesty="step13_honesty", intelligence="step14_intelligence",
)


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_csv(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _num(v):
    try:
        v = float(v)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _row(target, requirement, measured, met, source, note=""):
    return dict(target=target, requirement=requirement, measured=measured, met=met, source=source, note=note)


def _missing(target, requirement, source, why="evaluation output not found"):
    return _row(target, requirement, NOT_MEASURED, None, source, why)


def table_by_method(summary):
    """summary['table'] (list of row dicts from eval_sphota / eval_diffusion) -> {method: row}."""
    rows = _get(summary, "table") or []
    return {r.get("method"): r for r in rows if isinstance(r, dict)}


def ratio(num, den):
    """num / den, with inf when den is 0 and num > 0 (any faithful decode is infinitely better than none)."""
    num, den = _num(num), _num(den)
    if num is None or den is None:
        return None
    if den == 0:
        return math.inf if num > 0 else float("nan")
    return num / den


# --------------------------------------------------------------------------------------------- section 3.4 targets
def collect_targets(exp):
    exp = Path(exp)
    rows = []

    # 1. BD-rate vs JPEG (PSNR) on Kodak: at least 30% saving. Prefer the real-bitstream number of Step 8.
    s8, s7 = load_json(exp / PATHS["bitstream"] / "summary.json"), load_json(exp / PATHS["codec"] / "summary.json")
    bd, src = _num(_get(s8, "bd_rate_psnr_pct", "JPEG")), "step8_bitstream/summary.json"
    if bd is None:
        bd, src = _num(_get(s7, "bd_rate_psnr_pct", "JPEG")), "step7_codec/summary.json"
    rows.append(_row("BD-rate vs JPEG (PSNR), Kodak", ">= 30% bit saving", f"{bd:+.1f}%", bd <= -30, src,
                     "negative = fewer bits than JPEG; real file sizes" if "step8" in src else "estimated sizes")
                if bd is not None else _missing("BD-rate vs JPEG (PSNR), Kodak", ">= 30% bit saving", src,
                                                "needs all six lambda codecs evaluated"))

    # 2. LPIPS >= 25% lower than the VAE codec alone (same bits) and 3. FID below JPEG, WebP and the codec
    s10, sc = load_json(exp / PATHS["diffusion"] / "summary.json"), load_json(exp / PATHS["clic_crops"] / "summary.json")
    red = _num(_get(s10, "checks", "lpips_reduction_pct"))
    red_clic = _num(_get(sc, "checks", "lpips_reduction_pct"))
    note = f"CLIC crops ({_get(sc, 'crops')}): {red_clic:.1f}% lower" if red_clic is not None else "CLIC crop LPIPS not measured"
    rows.append(_row("LPIPS vs VAE codec alone", ">= 25% lower", f"{red:.1f}% lower (Kodak)", red >= 25,
                     "step10_diffusion/eval/summary.json", note)
                if red is not None else _missing("LPIPS vs VAE codec alone", ">= 25% lower", "step10_diffusion/eval/summary.json"))
    # FID: the guide measures it on about 2,000 CLIC crops; the 24-image Kodak FID is only a fallback and is labelled as such
    for src, label in ((sc, "step10_diffusion/clic_crops/summary.json"), (s10, "step10_diffusion/eval/summary.json")):
        fid = _get(src, "fid")
        if isinstance(fid, dict) and "diffusion" in fid:
            txt = ", ".join(f"{k} {v:.1f}" for k, v in fid.items())
            on = f"{_get(src, 'crops')} CLIC crops" if src is sc else "Kodak, 24 images (not the guide's CLIC crops)"
            rows.append(_row("FID on CLIC crops", "below JPEG, WebP and the VAE codec", f"{txt} [{on}]",
                             _get(src, "checks", "fid_beats_jpeg_webp_coarse"), label, _get(src, "fid_note") or ""))
            break
    else:
        rows.append(_missing("FID on CLIC crops", "below JPEG, WebP and the VAE codec", "step10_diffusion/clic_crops/summary.json",
                             "run scripts/eval_clic_crops.py --fid"))

    # 4. TFR >= 2x plain diffusion (text set), 5. FIR >= 1.5x plain diffusion (face set)
    for tname, kind, key, mult, setname in (("Text Faithfulness Rate", "tfr", "tfr", 2.0, "text"),
                                            ("Face Identity Rate", "fir", "fir", 1.5, "face")):
        path = exp / PATHS["guidance"] / f"eval_{setname}" / "summary.json"
        by = table_by_method(load_json(path))
        plain, full = _num(by.get("plain", {}).get(key)), _num(by.get("sphota", {}).get(key))
        req = f">= {mult:g}x plain diffusion"
        if plain is None or full is None:
            rows.append(_missing(tname, req, f"step12_guidance/eval_{setname}/summary.json", f"run eval_sphota.py --set {setname} --{key}"))
            continue
        r = ratio(full, plain)
        rows.append(_row(tname, req, f"Sphota {full:.1f}% vs plain {plain:.1f}% (x{r:.2f})" if math.isfinite(r) else
                         f"Sphota {full:.1f}% vs plain {plain:.1f}%", bool(r is not None and r >= mult),
                         f"step12_guidance/eval_{setname}/summary.json"))

    # 6. hallucination AUROC
    s13 = load_json(exp / PATHS["honesty"] / "summary.json")
    au = _num(_get(s13, "best", "auroc"))
    rows.append(_row("Hallucination detection AUROC", ">= 0.75", f"{au:.3f} ({_get(s13, 'best', 'detector')})", au >= 0.75,
                     "step13_honesty/summary.json", "best of the 8 detectors on held-out images") if au is not None
                else _missing("Hallucination detection AUROC", ">= 0.75", "step13_honesty/summary.json"))

    # 7. rate-control accuracy
    s14 = load_json(exp / PATHS["intelligence"] / "summary.json")
    rc, sp = _get(s14, "rate_control", "rate_control"), _get(s14, "rate_control", "size_prediction")
    if rc and _num(rc.get("within_target")) is not None:
        w = _num(rc["within_target"])
        note = (f"size predictor within +-10% for {100 * _num(sp.get('rows_within')):.1f}% of (image, mode) rows; "
                f"six discrete modes cap the target at {100 * _num(rc.get('oracle_within')):.1f}% (oracle)") if sp else ""
        rows.append(_row("Rate-control accuracy", "within +-10% of the target size for 80% of images",
                         f"{100 * w:.1f}% (meets-target {100 * _num(rc.get('meets')):.1f}%)", w >= 0.8,
                         "step14_intelligence/summary.json", note))
    else:
        rows.append(_missing("Rate-control accuracy", "within +-10% of the target size for 80% of images", "step14_intelligence/summary.json"))

    # 8. decode bit-exactness
    be = _get(s8, "all_bit_exact")
    rows.append(_row("Decode bit-exactness", "100% of test images", "100%" if be else "some images differ", bool(be),
                     "step8_bitstream/summary.json", f"{_get(s8, 'images_per_model')} images x {_get(s8, 'models')} models")
                if be is not None else _missing("Decode bit-exactness", "100% of test images", "step8_bitstream/summary.json"))
    df = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    df["met"] = df["met"].astype(object)             # True / False / None, never numpy bools or NaN
    return df


# --------------------------------------------------------------------------------------------- section 8.4 ablations
def _fmt(v, f="{:.3f}"):
    v = _num(v)
    return "n/a" if v is None else f.format(v)


def collect_ablations(exp):
    """One row per ablation A1-A8: question, status (done / not run) and a one-line headline taken from its output."""
    exp = Path(exp)
    rows = []

    def add(aid, name, status, headline, source):
        rows.append(dict(id=aid, ablation=name, status="done" if status else "not run", headline=headline if status else "", source=source))

    d = load_csv(exp / PATHS["codec"] / "ablation_k.csv")
    add("A1", "GMM K = 1 vs K = 3", d is not None and len(d),
        "" if d is None else f"K=3 needs {_fmt(d.saving_pct_equal_psnr.mean(), '{:+.1f}')}% bits at equal PSNR "
                             f"({_fmt(d.d_psnr_same_lambda.mean(), '{:+.2f}')} dB at equal lambda)", "step7_codec/ablation_k.csv")
    d = load_csv(exp / PATHS["codec"] / "ablation_act.csv")
    add("A2", "GDN vs ReLU", d is not None and len(d),
        "" if d is None else f"GDN needs {_fmt(d.saving_pct_equal_psnr.mean(), '{:+.1f}')}% bits at equal PSNR "
                             f"({_fmt(d.d_psnr_same_lambda.mean(), '{:+.2f}')} dB at equal lambda)", "step7_codec/ablation_act.csv")
    d = load_csv(exp / PATHS["diffusion"] / "sweep_cfg.csv")
    add("A3", "CFG scale w in {0, 1, 1.5, 2, 3}", d is not None and len(d),
        "" if d is None else "LPIPS " + ", ".join(f"w={r.w:g}: {_fmt(r.lpips)}" for r in d.itertuples()), "step10_diffusion/eval/sweep_cfg.csv")
    by = table_by_method(load_json(exp / PATHS["guidance"] / "eval_text" / "summary.json")) or \
        table_by_method(load_json(exp / PATHS["guidance"] / "eval_kodak" / "summary.json"))
    a4 = bool(by.get("plain") and by.get("guided_only"))
    add("A4", "with / without bin-consistency guidance", a4,
        "" if not a4 else f"consistency rate {_fmt(by['plain'].get('consistency'))} -> {_fmt(by['guided_only'].get('consistency'))}",
        "step12_guidance/eval_*/summary.json")
    a5 = bool(by.get("plain") and by.get("roi_only"))
    add("A5", "with / without ROI coding", a5,
        "" if not a5 else f"bpp {_fmt(by['plain'].get('bpp'))} -> {_fmt(by['roi_only'].get('bpp'))}; TFR {_fmt(by['plain'].get('tfr'), '{:.1f}')}% -> "
                          f"{_fmt(by['roi_only'].get('tfr'), '{:.1f}')}%", "step12_guidance/eval_*/summary.json")
    d = load_csv(exp / PATHS["diffusion"] / "sweep_steps.csv")
    add("A6", "DDIM steps {10, 25, 50}", d is not None and len(d),
        "" if d is None else "LPIPS " + ", ".join(f"{int(r.steps)} steps: {_fmt(r.lpips)}" for r in d.itertuples()),
        "step10_diffusion/eval/sweep_steps.csv")
    ab = load_csv(exp / PATHS["ablation"] / "summary.csv")
    for aid, name, run in (("A7", "linear vs cosine noise schedule", "linear_schedule"),
                           ("A8a", "Adam vs AdamW", "adam"), ("A8b", "constant vs cosine learning rate", "constant_lr")):
        ok = ab is not None and run in set(ab.run) and "base" in set(ab.run)
        h = ""
        if ok:
            b, r = ab[ab.run == "base"].iloc[0], ab[ab.run == run].iloc[0]
            h = f"LPIPS {_fmt(b.lpips, '{:.4f}')} (reference) vs {_fmt(r.lpips, '{:.4f}')} ({run})"
        add(aid, name, ok, h, "step10_ablation/summary.csv")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- faithfulness table
def faithfulness_rows(exp):
    """TFR / FIR for every decoding variant with 95% confidence intervals from the pass counts, plus the 8 detectors' AUROC."""
    exp = Path(exp)
    rows = []
    for setname, kind, num, den in (("text", "tfr", "n_pass", "n_eligible"), ("face", "fir", "n_pass", "n_faces")):
        s = load_json(exp / PATHS["guidance"] / f"eval_{setname}" / "summary.json")
        counts = _get(s, f"{kind}_counts") or {}
        for method, c in counts.items():
            k, n = c.get(num), c.get(den)
            if k is None or not n:
                continue
            lo, hi = binom_ci(int(k), int(n))
            rows.append(dict(metric=kind.upper(), set=setname, method=method, rate_pct=100 * k / n, ci_low=100 * lo, ci_high=100 * hi,
                             passed=int(k), n=int(n)))
    return pd.DataFrame(rows, columns=["metric", "set", "method", "rate_pct", "ci_low", "ci_high", "passed", "n"])


def auroc_rows(exp):
    s = load_json(Path(exp) / PATHS["honesty"] / "summary.json")
    a = _get(s, "auroc") or {}
    return pd.DataFrame(sorted(({"detector": k, "auroc": float(v)} for k, v in a.items()), key=lambda r: -r["auroc"]),
                        columns=["detector", "auroc"])


# --------------------------------------------------------------------------------------------------------- rendering
def to_markdown(df, floatfmt="{:.3f}"):
    if df is None or not len(df):
        return "_no data_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("")
            elif isinstance(v, bool):
                cells.append("yes" if v else "no")
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "/"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _as_flag(m):
    """True / False, or None when the target was not measured."""
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return None
    return bool(m)


def verdict_symbol(met):
    flag = _as_flag(met)
    return "MET" if flag is True else ("NOT MET" if flag is False else "not measured")


def targets_markdown(targets):
    flags = [_as_flag(m) for m in targets["met"]]
    t = targets.copy()
    t["met"] = [verdict_symbol(f) for f in flags]
    n_met, n_missing = sum(f is True for f in flags), sum(f is None for f in flags)
    head = f"{n_met} of {len(flags)} targets met" + (f", {n_missing} not measured yet" if n_missing else "") + ".\n\n"
    return head + to_markdown(t)
