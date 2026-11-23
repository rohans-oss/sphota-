"""PCA and k-means VQ course baselines on the RD plot next to JPEG / WebP (Step 4).

Trains the PCA bases and VQ codebooks on TRAINING images only, sweeps PCA step / VQ codebook size,
and measures bpp (ideal entropy bits / original pixels), PSNR, MS-SSIM, LPIPS on the test set.
Writes experiments/step4_baselines/rd_results.csv, report/figures/rd_kodak_all.png and
report/figures/pca_explained_variance.png.

    python scripts/eval_course_baselines.py                 # Kodak
    python scripts/eval_course_baselines.py --retrain       # ignore cached models
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.baselines.pca_codec import PCAImageCodec, sample_blocks  # noqa: E402
from truesight.baselines.vq_codec import VQCodec, sample_rgb_blocks  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.metrics import bd_rate, lpips_dist, msssim, psnr  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "step4_baselines"
FIGS = ROOT / "report" / "figures"
PCA_STEPS = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015]
PCA_KS = [16, 64]
VQ_CODES = [16, 64, 256, 1024]


def to_t(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def train_models(retrain, n_images, per_image):
    OUT.mkdir(parents=True, exist_ok=True)
    mdir = OUT / "models"
    mdir.mkdir(exist_ok=True)
    train = read_list(PROC / "train.txt")
    pca, vq = {}, {}
    pca_f = mdir / f"pca_n{n_images}_{per_image}.joblib"
    if pca_f.exists() and not retrain:
        pca = joblib.load(pca_f)
    else:
        t = time.time()
        L, C = sample_blocks(train, n_images, per_image)
        print(f"PCA training blocks: luma {L.shape}, chroma {C.shape}", flush=True)
        for k in PCA_KS:
            pca[k] = PCAImageCodec(k=k).fit(L, C)
        joblib.dump(pca, pca_f)
        print(f"  PCA trained in {time.time()-t:.0f}s", flush=True)
    vq_f = mdir / f"vq_n{n_images}_{per_image}.joblib"
    if vq_f.exists() and not retrain:
        vq = joblib.load(vq_f)
    else:
        B = sample_rgb_blocks(train, n_images, per_image)
        print(f"VQ training blocks: {B.shape}", flush=True)
        for n in VQ_CODES:
            t = time.time()
            vq[n] = VQCodec(n).fit(B)
            print(f"  k-means {n} codes in {time.time()-t:.0f}s", flush=True)
        joblib.dump(vq, vq_f)
    return pca, vq


def evaluate(files, pca, vq):
    rows = []
    for i, f in enumerate(files):
        ref = Image.open(f).convert("RGB")
        x, npx = to_t(ref), ref.width * ref.height
        jobs = [(f"PCA-{k}", step, lambda k=k, s=step: pca[k].code_image(ref, s)) for k in PCA_KS for step in PCA_STEPS]
        jobs += [("VQ", n, lambda n=n: vq[n].code_image(ref)) for n in VQ_CODES]
        for method, setting, fn in jobs:
            rec, bits = fn()
            y = to_t(rec)
            rows.append(dict(set="kodak", image=Path(f).name, method=method, quality=setting, bytes=bits / 8, bpp=bits / npx,
                             psnr=psnr(x, y), msssim=msssim(x, y), lpips=lpips_dist(x, y)))
        print(f"  {i+1}/{len(files)} {Path(f).name}", flush=True)
    return pd.DataFrame(rows)


def plot_rd(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s3 = ROOT / "experiments" / "step3_baselines" / "rd_results.csv"
    allv = pd.concat([pd.read_csv(s3).query("set == 'kodak'"), df], ignore_index=True)
    style = {"JPEG": ("tab:red", "o-"), "WEBP": ("tab:blue", "s-"), "PCA-16": ("tab:green", "^-"),
             "PCA-64": ("tab:olive", "v-"), "VQ": ("tab:purple", "D-")}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for m, (c, st) in style.items():
        g = allv[allv.method == m].groupby("quality").mean(numeric_only=True).sort_values("bpp")
        for a, col in zip(ax, ["psnr", "msssim", "lpips"]):
            a.plot(g.bpp, g[col], st, c=c, label=m, ms=4)
    for a, t in zip(ax, ["PSNR (dB)", "MS-SSIM", "LPIPS (lower = better)"]):
        a.set(xscale="log", xlabel="bits per pixel", ylabel=t, title=f"{t.split(' ')[0]} vs bpp - Kodak (24 images)")
        a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "rd_kodak_all.png", dpi=120)
    plt.close(fig)
    return allv


def plot_pca(pca):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = pca[64].luma.pca
    ev = p.explained_variance_ratio_
    cum = np.cumsum(ev)
    fig = plt.figure(figsize=(15, 4.6))
    a = fig.add_subplot(1, 3, 1)
    a.semilogy(np.arange(1, 65), ev, "o-", ms=3); a.set(xlabel="component", ylabel="explained variance ratio (log)", title="Scree plot (8x8 luma blocks)"); a.grid(alpha=.3)
    a = fig.add_subplot(1, 3, 2)
    a.plot(np.arange(1, 65), cum * 100, "o-", ms=3)
    for t in (90, 95, 99):
        n = int(np.searchsorted(cum, t / 100) + 1)
        a.axhline(t, c="gray", ls=":"); a.annotate(f"{t}% needs {n} comps", (n, t), xytext=(n + 3, t - 4), fontsize=8, arrowprops=dict(arrowstyle="-", color="gray"))
    a.set(xlabel="number of components kept", ylabel="cumulative explained variance (%)", title="Energy compaction"); a.grid(alpha=.3)
    g = fig.add_gridspec(4, 4, left=.69, right=.99, top=.9, bottom=.08, wspace=.08, hspace=.08)
    for i in range(16):
        b = fig.add_subplot(g[i // 4, i % 4]); b.imshow(p.components_[i].reshape(8, 8), cmap="gray"); b.axis("off")
    fig.text(.84, .94, "First 16 PCA basis blocks (DCT-like)", ha="center", fontsize=10)
    fig.savefig(FIGS / "pca_explained_variance.png", dpi=120)
    plt.close(fig)
    return {"components_for_90pct": int(np.searchsorted(cum, .90) + 1), "components_for_95pct": int(np.searchsorted(cum, .95) + 1),
            "components_for_99pct": int(np.searchsorted(cum, .99) + 1), "first_component_pct": float(ev[0] * 100),
            "top8_pct": float(cum[7] * 100), "top16_pct": float(cum[15] * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--train-images", type=int, default=200)
    ap.add_argument("--blocks-per-image", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    pca, vq = train_models(a.retrain, a.train_images, a.blocks_per_image)
    ev = plot_pca(pca)
    print("PCA explained variance:", ev)
    files = read_list(PROC / "test_kodak.txt")[: a.limit]
    df = evaluate(files, pca, vq)
    df.to_csv(OUT / "rd_results.csv", index=False)
    allv = plot_rd(df)
    mean = df.groupby(["method", "quality"]).mean(numeric_only=True).reset_index()
    print(mean[["method", "quality", "bpp", "psnr", "msssim", "lpips"]].round(4).to_string(index=False))
    bd = {}
    j = allv[allv.method == "JPEG"].groupby("quality").mean(numeric_only=True).sort_values("bpp")
    for m in ("PCA-16", "PCA-64", "VQ"):
        g = df[df.method == m].groupby("quality").mean(numeric_only=True).sort_values("bpp")
        try:
            bd[m] = round(bd_rate(j.bpp, j.psnr, g.bpp, g.psnr), 1)
        except Exception as e:
            bd[m] = f"n/a ({e})"
    print("BD-rate vs JPEG (PSNR, + = needs more bits):", bd)
    (OUT / "summary.json").write_text(json.dumps({"pca_variance": ev, "bd_rate_vs_jpeg_psnr": bd,
                                                  "points": {m: int(g.quality.nunique()) for m, g in df.groupby("method")}}, indent=2))


if __name__ == "__main__":
    main()
