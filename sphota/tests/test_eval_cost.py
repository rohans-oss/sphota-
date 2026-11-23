"""Evaluation cost limits, the CLIC crop evaluation (LPIPS + FID), the decode benchmark and the report skeleton."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from truesight import eval_limits as EL
from truesight.analysis import report as R
from truesight.analysis.report_doc import COURSE_MAP, DEVIATIONS, build_report, course_table, limitations
from truesight.codec.train_codec import train_codec
from truesight.codec.vae_codec import TrueSightCodec, bpp_terms
from truesight.diffusion import benchmark as BM
from truesight.diffusion.eval_crops import (build_summary, compute_fids, crop_positions, per_sample_bpp, plan_crops)
from truesight.diffusion.train_diffusion import train_diffusion
from truesight.sync import wanted

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ window limits
def test_window_size_keeps_aspect_ratio_and_budget():
    assert EL.window_size(768, 512, EL.MAX_PIXELS) == (768, 512)                  # already small enough
    assert EL.window_size(2048, 1365, 0) == (2048, 1365) and EL.window_size(2048, 1365, None) == (2048, 1365)
    w, h = EL.window_size(2048, 1365, EL.MAX_PIXELS)
    assert w % 16 == 0 and h % 16 == 0 and w * h <= EL.MAX_PIXELS and abs(w / h - 2048 / 1365) < 0.05
    assert w * h > 0.8 * EL.MAX_PIXELS                                              # not needlessly small
    w, h = EL.window_size(1000, 5000, 100_000)
    assert w * h <= 100_000 and h > w and w >= 64


def test_best_window_prefers_boxes_then_the_centre():
    assert EL.best_window(2000, 1000, 800, 500, []) == (592, 256)                  # no boxes: centred, corners on multiples of 16
    x, y = EL.best_window(2000, 1000, 800, 500, [(1700, 700, 1900, 800)])           # text in the bottom-right corner
    assert x + 800 >= 1900 and y + 500 >= 800 and x % 16 == 0 and y % 16 == 0
    x, y = EL.best_window(2000, 1000, 800, 500, [(10, 10, 100, 60), (1700, 900, 1900, 990)])   # two far apart: keeps the larger one
    assert x <= 1700 and x + 800 >= 1900 and y + 500 >= 990                          # the larger box wins
    assert EL.best_window(900, 600, 900, 600, [(5, 5, 50, 50)]) == (0, 0)             # window = image


class FakeDet:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img):
        return list(self.boxes)


def test_limit_and_detect_crops_to_the_boxes_and_shifts_them():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (1400, 2000, 3), dtype=np.uint8))
    text, faces = FakeDet([(1500, 900, 1800, 1000)]), FakeDet([(1520, 1100, 1600, 1180)])
    out, boxes, t, f, win = EL.limit_and_detect(img, text, faces, EL.MAX_PIXELS)
    x0, y0, x1, y1 = win
    assert out.size == (x1 - x0, y1 - y0) and out.size[0] * out.size[1] <= EL.MAX_PIXELS
    assert len(t) == 1 and len(f) == 1                                                # both detections are inside the window
    assert t[0] == (1500 - x0, 900 - y0, 1800 - x0, 1000 - y0)                        # shifted into the window's coordinates
    assert boxes and all(b[0] % 16 == 0 and b[1] % 16 == 0 and b[0] + b[2] <= out.size[0] and b[1] + b[3] <= out.size[1] for b in boxes)
    same, _, _, _, win2 = EL.limit_and_detect(img, text, None, 0)                     # no limit: untouched
    assert same is img and win2 == (0, 0, 2000, 1400)
    small = Image.new("RGB", (640, 480))
    assert EL.limit_and_detect(small, None, None)[0] is small


# ---------------------------------------------------------------------------------------------------- crop planning
def test_crop_positions_and_plan():
    pos = crop_positions(600, 700, 4, 256, seed=1)
    assert len(pos) == 4 and pos == crop_positions(600, 700, 4, 256, seed=1)
    assert all(y % 256 == 0 and x % 256 == 0 and y + 256 <= 600 and x + 256 <= 700 for y, x in pos) and len(set(pos)) == 4
    assert len(crop_positions(300, 300, 10, 256)) == 1 and crop_positions(100, 100, 5, 256) == []
    plan = plan_crops([(1024, 1024), (512, 768), (200, 200)], per_image=20, max_crops=2000)
    assert len(plan) == 16 + 2 * 3 and {i for i, _, _ in plan} == {0, 1}                # 16 + 6 windows exist, the 200 px image has none
    cut = plan_crops([(1024, 1024), (1024, 1024)], per_image=10, max_crops=6)
    assert len(cut) == 6 and {i for i, _, _ in cut} == {0, 1}                           # a small cap still spreads over the images
    assert plan == plan_crops([(1024, 1024), (512, 768), (200, 200)], 20, 2000)         # deterministic


def test_per_sample_bpp_matches_the_codec_rate():
    net = TrueSightCodec(N=16, M=24, K=3).eval()
    x = torch.rand(3, 3, 64, 64)
    with torch.no_grad():
        lik = net(x)["likelihoods"]
    per = per_sample_bpp(lik, 64 * 64)
    assert per.shape == (3,) and (per > 0).all()
    assert per.mean().item() == pytest.approx(sum(bpp_terms(lik, 3 * 64 * 64).values()).item(), rel=1e-4)


def test_fid_summary_with_a_fake_metric(tmp_path):
    for m in ("original", "coarse", "diffusion", "jpeg", "webp"):
        (tmp_path / m).mkdir()
    fid = compute_fids(tmp_path, fid_fn=lambda a, b: {"coarse": 90.0, "diffusion": 40.0, "jpeg": 120.0, "webp": 100.0}[b.name])
    assert fid == {"coarse": 90.0, "diffusion": 40.0, "jpeg": 120.0, "webp": 100.0}
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"bpp": rng.uniform(0.02, 0.05, 40), "lpips_coarse": rng.uniform(0.5, 0.6, 40), "lpips_diffusion": rng.uniform(0.3, 0.4, 40),
                         "lpips_jpeg": 0.7, "lpips_webp": 0.65, "reachable_jpeg": 0, "reachable_webp": 1})
    s = build_summary(rows, fid, dict(w=1.5))
    assert s["crops"] == 40 and s["checks"]["lpips_target_met"] and s["checks"]["fid_beats_jpeg_webp_coarse"]
    assert s["jpeg_reachable_fraction"] == 0.0 and "only 40 crops" in s["fid_note"] and s["w"] == 1.5
    assert "fid" not in build_summary(rows)


# ------------------------------------------------------------- the CLIC crop script end to end (tiny CPU models)
def _images(tmp_path, n=2, size=600):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        a = np.kron(rng.random((size // 8, size // 8, 3)), np.ones((8, 8, 1)))
        p = tmp_path / f"clic{i}.png"
        Image.fromarray((a * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    lst = tmp_path / "clic.txt"
    lst.write_text("\n".join(paths))
    return str(lst)


def test_clic_crop_script_pauses_resumes_and_recovers_lost_batches(tmp_path):
    lst = _images(tmp_path)
    kw = dict(K=3, N=16, M=24, batch=2, crop=64, workers=0, train_list=lst, device="cpu", log_every=1000, ckpt_every=1000)
    train_codec(0.0018, tmp_path / "codec", iters=1, **kw)
    train_diffusion(str(tmp_path / "codec" / "final.pt"), tmp_path / "diff", iters=1, batch=2, crop=64, channels=(16, 32, 32, 32), workers=0,
                    train_list=lst, device="cpu", amp=False, ckpt_every=1000)
    out = tmp_path / "crops"
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_clic_crops.py"), "--codec", str(tmp_path / "codec" / "final.pt"),
           "--diffusion", str(tmp_path / "diff" / "final.pt"), "--lists", lst, "--crops-per-image", "4", "--max-crops", "6", "--batch", "2",
           "--steps", "2", "--w", "1.5", "--out", str(out), "--device", "cpu"]
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT), TS_TIME_LIMIT_HOURS="0.00001", TS_SESSION_START="1")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 75, r.stdout[-1500:] + r.stderr[-1500:]                       # session time over: paused after the first batch
    assert len(pd.read_csv(out / "rows.csv")) == 2 and len(list((out / "batches").glob("*.npz"))) == 1
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    rows = pd.read_csv(out / "rows.csv")
    assert sorted(rows["crop"]) == list(range(6)) and rows["lpips_diffusion"].notna().all() and (rows["bpp"] > 0).all()
    s = json.loads((out / "summary.json").read_text())
    assert s["crops"] == 6 and {"coarse", "diffusion", "jpeg", "webp"} <= {t["method"] for t in s["table"]} and "lpips_reduction_pct" in s["checks"]
    assert not (out / "fid").exists()                                                    # PNG folders only exist while FID is computed
    victim = sorted((out / "batches").glob("*.npz"))[0]                                  # a batch file is lost (new machine, failed upload)
    victim.unlink()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert r.returncode == 0 and "already done; " in r.stdout
    assert sorted(pd.read_csv(out / "rows.csv")["crop"]) == list(range(6))               # the lost crops were recomputed, no hole, no duplicate


def test_sync_keeps_batches_but_not_png_folders():
    assert wanted("step10_diffusion/clic_crops/batches/00000.npz") and wanted("step10_diffusion/clic_crops/rows.csv")
    assert not wanted("step10_diffusion/clic_crops/fid/original/00001.png") and not wanted("x/fid/y.png")


# --------------------------------------------------------------------------------------------- benchmark
def test_benchmark_projection():
    secs = {"img": 0.1, "crop": 0.02}
    rows, total = BM.project(secs)
    names = [r[0] for r in rows]
    assert "clic_crops" in names and total == pytest.approx(sum(r[3] for r in rows))
    clic = next(r for r in rows if r[0] == "clic_crops")
    assert clic[3] == pytest.approx(2000 * 50 * 2 * 0.02 / 3600)
    small = BM.project(secs, BM.stage_plan(dict(text=10, face=10, kodak=2, crops=10, tune=1, hall=2, table=5, study=1, clic_images=2)))[1]
    assert small < total / 10
    from truesight.diffusion.unet import make_unet
    t = BM.time_unet_call(make_unet(64, (16, 32, 32, 32)).eval(), 64, 64, torch.device("cpu"), repeats=1)
    assert 0 < t < 30


# --------------------------------------------------------------------------------------------- report skeleton
def _results(tmp_path, with_clic=True):
    e = tmp_path / "exp"
    for sub in ("step10_diffusion/eval", "step10_diffusion/clic_crops"):
        (e / sub).mkdir(parents=True)
    (e / "step10_diffusion/eval/summary.json").write_text(json.dumps(
        dict(checks=dict(lpips_reduction_pct=31.0, fid_beats_jpeg_webp_coarse=False), fid=dict(diffusion=50.0, coarse=40.0, jpeg=60.0, webp=55.0))))
    if with_clic:
        (e / "step10_diffusion/clic_crops/summary.json").write_text(json.dumps(
            dict(crops=2000, checks=dict(lpips_reduction_pct=29.0, fid_beats_jpeg_webp_coarse=True),
                 fid=dict(diffusion=20.0, coarse=30.0, jpeg=60.0, webp=55.0))))
    return e


def test_fid_target_uses_the_clic_crops_and_labels_the_kodak_fallback(tmp_path):
    t = R.collect_targets(_results(tmp_path))
    row = t[t.target == "FID on CLIC crops"].iloc[0]
    assert row["met"] is True and "2000 CLIC crops" in row["measured"]
    assert "CLIC crops (2000): 29.0% lower" in t[t.target == "LPIPS vs VAE codec alone"].iloc[0]["note"]
    t2 = R.collect_targets(_results(tmp_path / "b", with_clic=False))
    row2 = t2[t2.target == "FID on CLIC crops"].iloc[0]
    assert row2["met"] is False and "Kodak, 24 images" in row2["measured"]
    assert "eval_clic_crops" in R.collect_targets(tmp_path / "none").query("target == 'FID on CLIC crops'").iloc[0]["note"]


def test_course_map_points_at_files_that_exist():
    assert len(COURSE_MAP) == 28
    table = course_table(ROOT)
    assert "MISSING" not in table, table
    assert len(DEVIATIONS) >= 8


def test_report_skeleton_marks_what_is_not_generated(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    targets = R.collect_targets(_results(tmp_path))
    targets.to_csv(res / "summary_table.csv", index=False)
    (res / "summary_table.md").write_text(R.targets_markdown(targets))
    abl = R.collect_ablations(tmp_path / "exp")
    abl.to_csv(res / "ablation_table.csv", index=False)
    (res / "ablation_table.md").write_text(R.to_markdown(abl))
    (res / "provenance.json").write_text(json.dumps(dict(commit="abcdef1234567", frozen=False, generated="now")))
    md = build_report(ROOT, res)
    assert md.count("TODO(author):") == 6 and "NOT frozen" in md and "abcdef1234" in md
    assert "Success targets" in md and "Differences from the build guide" in md and "not measured" in md
    lim = limitations(targets, abl)
    assert any("not measured" in x for x in lim) and any("Ablation A1" in x for x in lim)
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "build_report.py"), "--results", str(res), "--out", str(tmp_path / "out" / "report.md"),
                        "--root", str(ROOT)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1000:]
    assert (tmp_path / "out" / "report.md").exists() and "Week" in (tmp_path / "out" / "weekly_log.md").read_text()
