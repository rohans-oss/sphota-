"""Measure how much extra training buys: the same decode on the same Kodak images with several checkpoints of one run.

    python scripts/compare_checkpoints.py --codec experiments/step7_codec/lmbda0.0018/final.pt \
        --checkpoints experiments/step10_diffusion/snapshot_30000.pt experiments/step10_diffusion/final.pt \
        --images 30 --steps 25 --out experiments/step10_diffusion/checkpoint_compare

Every checkpoint decodes the identical latents with the same seed, sampler steps and guidance scale. The codec-only (coarse) decode is
the reference row. Writes <out>/per_image.csv and summary.csv (mean LPIPS / PSNR per checkpoint, change vs the first checkpoint, and a
paired bootstrap 95% CI of the LPIPS difference to the first checkpoint) and prints the table.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.ablation import evaluate_unet  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(codec, checkpoints, files, steps=25, w=1.5, images=30, crop=None, device="cpu"):
    """Returns (per-image DataFrame of every checkpoint, summary DataFrame)."""
    dev = torch.device(device)
    frames = {}
    for ck in checkpoints:
        unet, cfg = load_unet(ck, dev)
        step = int(torch.load(ck, map_location="cpu")["step"])
        df = evaluate_unet(unet, Schedule(cfg["T"], cfg["kind"], device=dev), codec, files, steps=steps, w=w, device=dev,
                           max_images=images, crop=crop)
        frames[f"step {step}"] = df.assign(checkpoint=f"step {step}", step=step)
        print(f"step {step}: LPIPS {df.lpips.mean():.4f}  PSNR {df.psnr.mean():.2f}", flush=True)
    names = list(frames)
    base = frames[names[0]].set_index("image")
    rows = [dict(checkpoint="codec only (no diffusion)", step=0, images=len(base), lpips=float(base.lpips_coarse.mean()),
                 psnr=float(base.psnr_coarse.mean()))]
    for n in names:
        d = frames[n].set_index("image")
        row = dict(checkpoint=n, step=int(d.step.iloc[0]), images=len(d), lpips=float(d.lpips.mean()), psnr=float(d.psnr.mean()))
        if n != names[0]:
            diff = (d.lpips - base.lpips).to_numpy()
            lo, hi = boot_ci(diff)
            row.update(lpips_change_vs_first=float(diff.mean()), ci95_low=lo, ci95_high=hi,
                       psnr_change_vs_first=float((d.psnr - base.psnr).mean()))
        rows.append(row)
    return pd.concat(frames.values(), ignore_index=True), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", default=str(ROOT / "experiments" / "step7_codec" / "lmbda0.0018" / "final.pt"))
    ap.add_argument("--checkpoints", nargs="+", required=True, help="two or more U-Net checkpoints, oldest first")
    ap.add_argument("--test-list", default=str(ROOT / "data" / "processed" / "test_kodak.txt"))
    ap.add_argument("--images", type=int, default=30)
    ap.add_argument("--crop", type=int, default=None, help="centre-crop the images (CPU runs)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step10_diffusion" / "checkpoint_compare"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    codec, _ = load_codec(a.codec, torch.device(a.device))
    per, summ = compare(codec, a.checkpoints, read_list(a.test_list), a.steps, a.w, a.images, a.crop, a.device)
    per.to_csv(out / "per_image.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    print("\n" + summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nLower LPIPS is better. A CI that excludes 0 means the later checkpoint is measurably different on these images.")


if __name__ == "__main__":
    main()
