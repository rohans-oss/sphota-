"""Summary table of the section 3.4 targets, ablation status and faithfulness tables (truesight/analysis/report.py)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from truesight.analysis import report as R

ROOT = Path(__file__).resolve().parents[1]


def _w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _full_experiments(tmp_path, tfr=(40.0, 10.0), fir=(90.0, 50.0), bd=-35.0, auroc=0.8, within=0.85):
    e = tmp_path / "exp"
    _w(e / "step8_bitstream" / "summary.json", dict(all_bit_exact=True, images_per_model=24, models=6, bd_rate_psnr_pct={"JPEG": bd}))
    _w(e / "step10_diffusion/eval" / "summary.json", dict(checks=dict(lpips_reduction_pct=31.0, fid_beats_jpeg_webp_coarse=True),
                                                         fid=dict(diffusion=20.0, coarse=30.0, jpeg=60.0, webp=55.0), fid_note="n=24"))
    for setname, key, (full, plain), counts in (("text", "tfr", tfr, dict(n_eligible=100)), ("face", "fir", fir, dict(n_faces=200))):
        n = list(counts.values())[0]
        _w(e / "step12_guidance" / f"eval_{setname}" / "summary.json", dict(
            table=[dict(method="plain", bpp=0.05, consistency=0.6, **{key: plain}), dict(method="roi_only", bpp=0.07, **{key: plain + 5}),
                   dict(method="guided_only", consistency=0.95, **{key: plain + 2}), dict(method="sphota", bpp=0.07, **{key: full})],
            **{f"{key}_counts": {m: dict(n_pass=int(v * n / 100), **counts) for m, v in (("plain", plain), ("sphota", full))}}))
    _w(e / "step13_honesty" / "summary.json", dict(best=dict(detector="xgboost", auroc=auroc),
                                                  auroc={"xgboost": auroc, "logistic": auroc - 0.1}))
    _w(e / "step14_intelligence" / "summary.json", dict(rate_control=dict(
        size_prediction=dict(rows_within=0.9), rate_control=dict(within_target=within, meets=0.97, oracle_within=0.92))))
    for sub in ("step7_codec", "step10_diffusion/eval", "step10_ablation"):
        (e / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(lmbda=[0.0130], saving_pct_equal_psnr=[-8.0], d_psnr_same_lambda=[0.3])).to_csv(e / "step7_codec" / "ablation_k.csv", index=False)
    pd.DataFrame(dict(w=[0, 1.5], lpips=[0.4, 0.3])).to_csv(e / "step10_diffusion/eval" / "sweep_cfg.csv", index=False)
    pd.DataFrame(dict(steps=[10, 25], lpips=[0.35, 0.3])).to_csv(e / "step10_diffusion/eval" / "sweep_steps.csv", index=False)
    pd.DataFrame(dict(run=["base", "linear_schedule", "adam", "constant_lr"], lpips=[0.30, 0.33, 0.31, 0.34])).to_csv(
        e / "step10_ablation" / "summary.csv", index=False)
    return e


def test_all_eight_targets_met(tmp_path):
    t = R.collect_targets(_full_experiments(tmp_path))
    assert len(t) == 8 and t["met"].tolist() == [True] * 8
    assert "x4.00" in t.loc[t.target == "Text Faithfulness Rate", "measured"].iloc[0]
    assert R.targets_markdown(t).startswith("8 of 8 targets met.")


def test_missed_targets_are_reported_as_not_met(tmp_path):
    t = R.collect_targets(_full_experiments(tmp_path, tfr=(15.0, 10.0), fir=(60.0, 50.0), bd=-12.0, auroc=0.7, within=0.5))
    met = dict(zip(t.target, t["met"]))
    assert met["BD-rate vs JPEG (PSNR), Kodak"] is False and met["Text Faithfulness Rate"] is False
    assert met["Face Identity Rate"] is False and met["Hallucination detection AUROC"] is False
    assert met["Rate-control accuracy"] is False and met["LPIPS vs VAE codec alone"] is True


def test_nothing_measured_is_never_a_pass_or_a_fail(tmp_path):
    t = R.collect_targets(tmp_path / "empty")
    assert len(t) == 8 and (t["measured"] == R.NOT_MEASURED).all()
    assert all(m is None for m in t["met"])
    assert "0 of 8 targets met, 8 not measured yet" in R.targets_markdown(t)


def test_partial_results(tmp_path):
    e = tmp_path / "e"
    _w(e / "step8_bitstream" / "summary.json", dict(all_bit_exact=False, images_per_model=24, models=6))
    t = R.collect_targets(e)
    row = t[t.target == "Decode bit-exactness"].iloc[0]
    assert row["met"] is False and "differ" in row["measured"]
    assert t[t.target == "BD-rate vs JPEG (PSNR), Kodak"].iloc[0]["measured"] == R.NOT_MEASURED


def test_tfr_ratio_when_plain_decoding_never_passes(tmp_path):
    t = R.collect_targets(_full_experiments(tmp_path, tfr=(30.0, 0.0)))
    row = t[t.target == "Text Faithfulness Rate"].iloc[0]
    assert row["met"] is True                                       # anything above zero is infinitely better than none
    assert R.ratio(5, 0) == math.inf and math.isnan(R.ratio(0, 0)) and R.ratio(None, 1) is None


def test_ablation_status(tmp_path):
    a = R.collect_ablations(_full_experiments(tmp_path)).set_index("id")
    assert list(a.index) == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8a", "A8b"]
    assert a.loc["A1", "status"] == "done" and a.loc["A2", "status"] == "not run"
    assert a.loc[["A3", "A4", "A5", "A6", "A7", "A8a", "A8b"], "status"].eq("done").all()
    assert "0.600 -> 0.950" in a.loc["A4", "headline"] and "reference" in a.loc["A7", "headline"]
    assert (R.collect_ablations(tmp_path / "nothing")["status"] == "not run").all()


def test_faithfulness_rows_have_confidence_intervals(tmp_path):
    f = R.faithfulness_rows(_full_experiments(tmp_path))
    r = f[(f.metric == "TFR") & (f.method == "sphota")].iloc[0]
    assert r.n == 100 and r.passed == 40 and r.rate_pct == pytest.approx(40.0)
    assert r.ci_low < 40.0 < r.ci_high and r.ci_low == pytest.approx(30.3, abs=0.5)
    a = R.auroc_rows(_full_experiments(tmp_path))
    assert a.detector.tolist() == ["xgboost", "logistic"]
    assert R.faithfulness_rows(tmp_path / "x").empty


def test_make_figures_end_to_end(tmp_path):
    e = _full_experiments(tmp_path)
    out, figs = tmp_path / "results", tmp_path / "figs"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "make_figures.py"), "--exp", str(e), "--out", str(out),
                        "--figs", str(figs), "--study", str(tmp_path / "nostudy")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    for f in ("summary_table.csv", "summary_table.md", "ablation_table.md", "faithfulness.csv", "provenance.json"):
        assert (out / f).exists(), f
    assert (figs / "targets_overview.png").stat().st_size > 1000
    assert (out / "step13_honesty" / "summary.json").exists()                     # per-step results are copied along
    assert "8 of 8 targets met" in (out / "summary_table.md").read_text()
    r2 = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "make_figures.py"), "--exp", str(e), "--out", str(out),
                         "--figs", str(figs), "--require-freeze", "--no-copy"], capture_output=True, text=True, timeout=300)
    assert r2.returncode != 0 and "eval-freeze" in (r2.stderr + r2.stdout)         # not a tagged, clean commit


def test_eval_faithfulness_script(tmp_path):
    e = _full_experiments(tmp_path)
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "eval_faithfulness.py"), "--exp", str(e), "--out", str(tmp_path / "o"),
                        "--figs", str(tmp_path / "f")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "o" / "faithfulness.md").read_text().count("sphota") >= 2 and (tmp_path / "f" / "faithfulness.png").exists()
