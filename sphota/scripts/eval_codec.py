"""Step 7 evaluation: codec RD curve on Kodak vs JPEG/WebP/PCA/VQ, BD-rate table, K=1 vs K=3 ablation.

    python scripts/eval_codec.py                          # uses experiments/step7_codec/{lmbda*,K1_lmbda*}/final.pt
    python scripts/eval_codec.py --exp experiments/step7_codec --limit 4     # quick check on 4 images

Reads the baseline curves written by Steps 3-4 (experiments/step3_baselines, step4_baselines). Writes into --exp:
rd_results.csv (per image), rd_points.csv (per lambda, mean over images), summary.json (BD-rates, 'beats JPEG below
0.5 bpp' check, ablation, NaN check) and ablation_k.csv, plus report/figures/rd_kodak_codec.png and
codec_training_curves.png. bpp is the estimated rate (ideal arithmetic coder); real files come in Step 8.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from truesight.codec.eval_codec import (ablation_act, ablation_k, bd_table, beats_baseline, evaluate_codec,  # noqa: E402
                                        mean_point)
from truesight.codec.train_codec import load_codec  # noqa: E402

PROC = ROOT / "data" / "processed"
FIGS = ROOT / "report" / "figures"


def find_runs(exp):
    runs = []
    for p in sorted(Path(exp).glob("*lmbda*/final.pt")):
        if p.parent.name.startswith("relu_"):                         # ablation A2 runs are evaluated separately
            continue
        K = 1 if p.parent.name.startswith("K1_") else 3
        runs.append((K, p))
    return runs


def find_relu_runs(exp):
    return sorted(Path(exp).glob("relu_lmbda*/final.pt"))


def baseline_curves():
    """Kodak mean RD points of JPEG, WebP (Step 3) and PCA-16/64, VQ (Step 4), keyed by method name."""
    frames = [pd.read_csv(ROOT / "experiments" / "step3_baselines" / "rd_results.csv").query("set == 'kodak'")]
    s4 = ROOT / "experiments" / "step4_baselines" / "rd_results.csv"
    if s4.exists():
        frames.append(pd.read_csv(s4).query("set == 'kodak'"))
    allv = pd.concat(frames, ignore_index=True)
    return {m: g.groupby("quality").mean(numeric_only=True).reset_index().sort_values("bpp")
            for m, g in allv.groupby("method")}


def plot_rd(points, base, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"JPEG": ("tab:red", "o-"), "WEBP": ("tab:blue", "s-"), "PCA-16": ("tab:green", "^-"),
             "PCA-64": ("tab:olive", "v-"), "VQ": ("tab:purple", "D-")}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for m, g in base.items():
        c, st = style.get(m, ("gray", "x-"))
        for a, col in zip(ax, ["psnr", "msssim", "lpips"]):
            if col in g:
                a.plot(g.bpp, g[col], st, c=c, label=m, ms=3, alpha=.8)
    for K, c, st in ((3, "k", "*-"), (1, "tab:orange", "x--")):
        g = points[points.K == K].sort_values("bpp")
        for a, col in zip(ax, ["psnr", "msssim", "lpips"]):
            if len(g) and col in g:
                a.plot(g.bpp, g[col], st, c=c, label=f"TrueSight K={K}", ms=8 if K == 3 else 6, lw=2)
    for a, t in zip(ax, ["PSNR (dB)", "MS-SSIM", "LPIPS (lower = better)"]):
        a.set(xscale="log", xlabel="estimated bits per pixel", ylabel=t, title=f"{t.split(' ')[0]} vs bpp - Kodak"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_training(exp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logs = {p.parent.name: pd.read_csv(p) for p in sorted(Path(exp).glob("*lmbda*/log.csv")) if p.stat().st_size > 0}
    logs = {n: d for n, d in logs.items() if len(d)}              # runs shorter than --log-every have a header only
    if not logs:
        return {}
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    for name, d in logs.items():
        for a, col in zip(ax, ["loss", "bpp", "mse", "aux"]):
            a.plot(d["it"], d[col], label=name, lw=1)
    for a, t in zip(ax, ["RD loss", "bpp (training batch)", "MSE", "aux loss"]):
        a.set(xlabel="iteration", title=t); a.grid(alpha=.3)
    ax[2].set_yscale("log"); ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return {n: bool(np.isfinite(d[["loss", "bpp", "mse", "aux"]].to_numpy()).all()) for n, d in logs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=str(ROOT / "experiments" / "step7_codec"))
    ap.add_argument("--test-list", default=str(PROC / "test_kodak.txt"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--figs", default=str(FIGS))
    ap.add_argument("--no-perceptual", action="store_true", help="skip MS-SSIM / LPIPS (faster)")
    a = ap.parse_args()
    exp, figs = Path(a.exp), Path(a.figs); figs.mkdir(parents=True, exist_ok=True)
    runs = find_runs(exp)
    if not runs:
        sys.exit(f"no final.pt found under {exp}; train with scripts/train_all_lambdas.py first")

    rows, pts = [], []
    for K, p in runs:
        net, cfg = load_codec(p)
        df = evaluate_codec(net, a.test_list, max_images=a.limit, with_perceptual=not a.no_perceptual)
        df.insert(0, "K", K); df.insert(1, "lmbda", cfg["lmbda"])
        rows.append(df)
        pt = mean_point(df.drop(columns=["K", "lmbda"]), K=K, lmbda=cfg["lmbda"])
        pts.append(pt)
        print(f"K={K} lmbda={cfg['lmbda']:.4f}: {pt['bpp']:.4f} bpp  {pt['psnr']:.2f} dB", flush=True)
    per_image, points = pd.concat(rows, ignore_index=True), pd.DataFrame(pts).sort_values(["K", "lmbda"])
    per_image.to_csv(exp / "rd_results.csv", index=False); points.to_csv(exp / "rd_points.csv", index=False)

    base = baseline_curves()
    k3 = points[points.K == 3]
    summary = {"n_models": {"K3": int((points.K == 3).sum()), "K1": int((points.K == 1).sum())},
               "finite_rd": bool(np.isfinite(points[["bpp", "psnr"]].to_numpy()).all())}
    if len(k3) >= 4:
        summary["bd_rate_psnr_pct"] = {k: round(v, 1) for k, v in bd_table(k3, base, "psnr").items()}
        if "msssim" in k3:
            summary["bd_rate_msssim_pct"] = {k: round(v, 1) for k, v in bd_table(k3, base, "msssim").items()}
        v = summary["bd_rate_psnr_pct"].get("JPEG")
        summary["target_bd_rate_vs_jpeg_le_-30"] = None if v is None or np.isnan(v) else bool(v <= -30)
    else:
        summary["note"] = "BD-rate needs >= 4 K=3 points (train all six lambdas)"
    if "JPEG" in base and len(k3):
        ok, det = beats_baseline(k3, base["JPEG"], 0.5)
        summary["beats_jpeg_below_0.5bpp"] = ok
        summary["beats_jpeg_details"] = det
    k1 = points[points.K == 1]
    if len(k1) and len(k3) >= 2:
        ab = ablation_k(k3, k1)
        ab.to_csv(exp / "ablation_k.csv", index=False)
        summary["ablation_k1_vs_k3"] = ab.round(4).to_dict("records")
        print("\nK=1 vs K=3:\n" + ab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    relu_paths = find_relu_runs(exp)
    if relu_paths and len(k3):
        rp = []
        for p in relu_paths:
            net, cfg = load_codec(p)
            df = evaluate_codec(net, a.test_list, max_images=a.limit, with_perceptual=not a.no_perceptual)
            rp.append(mean_point(df, lmbda=cfg["lmbda"]))
            print(f"ReLU lmbda={cfg['lmbda']:.4f}: {rp[-1]['bpp']:.4f} bpp  {rp[-1]['psnr']:.2f} dB", flush=True)
        a2 = ablation_act(k3, pd.DataFrame(rp))
        a2.to_csv(exp / "ablation_act.csv", index=False)
        summary["ablation_a2_relu_vs_gdn"] = a2.round(4).to_dict("records")
        print("\nReLU vs GDN (A2):\n" + a2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    plot_rd(points, base, figs / "rd_kodak_codec.png")
    summary["training_logs_finite"] = plot_training(exp, figs / "codec_training_curves.png")
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: None if o != o else float(o)))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "beats_jpeg_details"}, indent=2, default=str))


if __name__ == "__main__":
    main()
