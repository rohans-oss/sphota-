"""Step 6: plain autoencoder baseline. Trains PlainAE with MSE only for several bottleneck sizes M, then
reports PSNR and the *measured* bits (entropy of the rounded latents) on Kodak.

    python scripts/train_ae.py                                  # guide settings: M=8,16,32, 30k it, batch 16, 256 crops
    python scripts/train_ae.py --iters 2000 --batch 8 --crop 128 --N 64   # reduced CPU run
    python scripts/train_ae.py --M 16 --seeds 0 1 2             # same M, different seeds: how much do the bits move?

Resumable: re-running continues from experiments/step6_ae/M*_s*/ckpt.pt.
Outputs (under --out): results.csv / results.json, per-run train_log.csv, and
report/figures/ae_samples.png (Kodak originals vs reconstructions) + ae_bits_vs_M.png.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402  (torch before pandas/sklearn: avoids a Windows DLL clash)
import pandas as pd  # noqa: E402
from truesight.codec.train_ae import evaluate_ae, train_ae  # noqa: E402
from truesight.data import FullImageDataset  # noqa: E402

PROC = ROOT / "data" / "processed"
FIGS = ROOT / "report" / "figures"


def sample_figure(rows, test_list, idxs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = FullImageDataset(test_list, multiple=16)
    fig, ax = plt.subplots(len(idxs), 1 + len(rows), figsize=(3.2 * (1 + len(rows)), 2.4 * len(idxs)), squeeze=False)
    for r, i in enumerate(idxs):
        x, (h, w), _ = ds[i]
        ax[r][0].imshow(x[:, :h, :w].permute(1, 2, 0).numpy()); ax[r][0].set_title("original" if r == 0 else "")
        for c, row in enumerate(rows, 1):
            ax[r][c].imshow(row["recons"][i].permute(1, 2, 0).numpy())
            if r == 0:
                ax[r][c].set_title(f"M={row['M']}  {row['psnr']:.1f} dB  {row['bpp']:.3f} bpp", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def bits_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for s, g in df.groupby("seed"):
        ax[0].plot(g.M, g.bpp, "o-", label=f"seed {s}"); ax[1].plot(g.bpp, g.psnr, "o-", label=f"seed {s}")
        for _, r in g.iterrows():
            ax[1].annotate(f"M={int(r.M)}", (r.bpp, r.psnr), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[0].set(xlabel="bottleneck channels M", ylabel="measured bpp (entropy of rounded latents)",
              title="Bits are a by-product of training, not a setting"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set(xlabel="measured bpp", ylabel="Kodak PSNR (dB)", title="PlainAE: PSNR vs measured bpp"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--N", type=int, default=128, help="hidden width")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--train-list", default=str(PROC / "train.txt"))
    ap.add_argument("--test-list", default=str(PROC / "test_kodak.txt"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step6_ae"))
    ap.add_argument("--figs", default=str(FIGS))
    ap.add_argument("--max-test", type=int, default=None, help="limit test images (debug only)")
    a = ap.parse_args()
    out, figs = Path(a.out), Path(a.figs); out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in a.seeds:
        for M in a.M:
            run = out / f"M{M}_s{seed}"
            model, _ = train_ae(a.train_list, run, M=M, N=a.N, iters=a.iters, batch=a.batch, crop=a.crop, lr=a.lr,
                                seed=seed, workers=a.workers)
            res, recons = evaluate_ae(model, a.test_list, max_images=a.max_test)
            res.update(M=M, seed=seed, N=a.N, iters=a.iters, batch=a.batch, crop=a.crop)
            rows.append({**res, "recons": recons})
            print(f"== M={M} seed={seed}: PSNR {res['psnr']:.2f} dB, {res['bpp']:.4f} bpp, "
                  f"latent range [{res['latent_min']}, {res['latent_max']}], {res['latent_unique']} symbols", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "recons"} for r in rows])
    cols = ["M", "seed", "psnr", "bpp", "bits_per_image_mean", "latent_min", "latent_max", "latent_unique",
            "latent_mean_abs", "latent_elems_per_pixel", "n_images", "N", "iters", "batch", "crop"]
    df = df[cols]; df.to_csv(out / "results.csv", index=False)
    (out / "results.json").write_text(json.dumps(df.to_dict("records"), indent=2))
    print("\n" + df[["M", "seed", "psnr", "bpp", "bits_per_image_mean", "latent_min", "latent_max", "latent_unique"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    first = [r for r in rows if r["seed"] == a.seeds[0]]
    sample_figure(first, a.test_list, list(range(0, first[0]["n_images"], max(1, first[0]["n_images"] // 4)))[:4],
                  figs / "ae_samples.png")
    bits_figure(df, figs / "ae_bits_vs_M.png")
    print(f"saved {out/'results.csv'}, {figs/'ae_samples.png'}, {figs/'ae_bits_vs_M.png'}")


if __name__ == "__main__":
    from truesight import runtime
    runtime.run_cli(main)
