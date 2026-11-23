"""Rate-distortion evaluation of the JPEG / WebP baselines (Step 3).

For every test image and quality setting: encode, measure bpp from the real byte count, then
PSNR, MS-SSIM and LPIPS. Saves one CSV row per (set, image, method, setting), mean RD curves,
and the smallest file size JPEG can reach.

    python scripts/eval_rd.py                                   # Kodak only
    python scripts/eval_rd.py --sets kodak test_clic test_text test_face --limit 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.baselines.jpeg import bpp_of, find_quality_for_bpp, min_file_size, pil_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.metrics import lpips_dist, msssim, psnr  # noqa: E402

PROC = ROOT / "data" / "processed"
SETTINGS = {"JPEG": [2, 5, 10, 20, 30, 50, 70, 90], "WEBP": [1, 5, 10, 25, 50, 75, 90]}
LISTS = {"kodak": "test_kodak", "clic": "test_clic", "text": "test_text", "face": "test_face"}


def to_t(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def evaluate_set(name: str, files: list[str]) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        ref = Image.open(f).convert("RGB")
        x = to_t(ref)
        for fmt, qs in SETTINGS.items():
            for q in qs:
                rec, n = pil_codec(ref, q, fmt)
                y = to_t(rec)
                rows.append(dict(set=name, image=Path(f).name, method=fmt, quality=q, bytes=n,
                                 bpp=bpp_of(n, ref), psnr=psnr(x, y), msssim=msssim(x, y), lpips=lpips_dist(x, y)))
        if (i + 1) % 5 == 0 or i + 1 == len(files):
            print(f"  [{name}] {i+1}/{len(files)} images  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for (m, c) in (("JPEG", "tab:red"), ("WEBP", "tab:blue")):
        g = df[df.method == m].groupby("quality").mean(numeric_only=True).sort_values("bpp")
        for a, col, lab in zip(ax, ["psnr", "msssim", "lpips"], ["PSNR (dB)", "MS-SSIM", "LPIPS (lower = better)"]):
            a.plot(g.bpp, g[col], "o-", c=c, label=m)
            for q, r in g.iterrows():
                a.annotate(str(q), (r.bpp, r[col]), fontsize=6, xytext=(2, 3), textcoords="offset points")
            a.set(xlabel="bits per pixel (real file size)", ylabel=lab, title=f"{lab.split(' ')[0]} vs bpp - {tag}")
            a.grid(alpha=.3)
    ax[0].legend()
    for a in ax:
        a.set_xscale("log")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"rd_{tag}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["kodak"], choices=list(LISTS))
    ap.add_argument("--limit", type=int, default=None, help="max images per set (Kodak always uses all 24)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step3_baselines"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = ROOT / "report" / "figures"
    torch.set_num_threads(max(1, torch.get_num_threads()))

    frames, summary = [], {}
    for s in a.sets:
        files = read_list(PROC / f"{LISTS[s]}.txt")
        if a.limit and s != "kodak":
            files = files[: a.limit]
        print(f"== {s}: {len(files)} images", flush=True)
        df = evaluate_set(s, files)
        frames.append(df)
        # ---- size floor: smallest file each codec can write
        floors = {}
        for fmt in ("JPEG", "WEBP"):
            m = [min_file_size(Image.open(f).convert("RGB"), fmt) for f in files]
            floors[fmt] = {"min_bytes_mean": float(np.mean([v[0] for v in m])), "min_bpp_mean": float(np.mean([v[1] for v in m])),
                           "min_bpp_min": float(np.min([v[1] for v in m])), "min_bpp_max": float(np.max([v[1] for v in m])),
                           "quality_at_floor": m[0][2]}
        # ---- binary search demo: can JPEG hit these target bpp?
        search = {}
        for t in (0.05, 0.1, 0.2, 0.5, 1.0):
            res = [find_quality_for_bpp(Image.open(f).convert("RGB"), t, "JPEG") for f in files[:min(len(files), 24)]]
            search[str(t)] = {"mean_bpp_achieved": float(np.mean([r[1] for r in res])),
                              "mean_quality": float(np.mean([r[0] for r in res])),
                              "fraction_reachable": float(np.mean([r[3] for r in res]))}
        mean = df.groupby(["method", "quality"]).mean(numeric_only=True).reset_index()
        summary[s] = {"images": len(files), "size_floor": floors, "binary_search_jpeg": search,
                      "mean_curves": mean[["method", "quality", "bpp", "psnr", "msssim", "lpips"]].round(4).to_dict("records"),
                      "plot": str(plot(df, figs, s).relative_to(ROOT))}
        print(f"   JPEG size floor ({s}): {floors['JPEG']['min_bpp_mean']:.3f} bpp mean "
              f"(range {floors['JPEG']['min_bpp_min']:.3f}-{floors['JPEG']['min_bpp_max']:.3f}) at quality {floors['JPEG']['quality_at_floor']}")
        print(f"   WebP size floor ({s}): {floors['WEBP']['min_bpp_mean']:.3f} bpp mean")
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out / "rd_results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {len(full)} rows -> {out/'rd_results.csv'}")


if __name__ == "__main__":
    main()
