"""Step 8 evaluation: real .sphota files on Kodak - RD curve from real file sizes, bit-exactness, timing.

    python scripts/eval_bitstream.py                         # all K=3 checkpoints in experiments/step7_codec
    python scripts/eval_bitstream.py --limit 3               # quick check on 3 images
    python scripts/eval_bitstream.py --device cpu --save-all # also keep every .sphota file

torchac compiles a C++ extension, so on Windows run this from an "x64 Native Tools" prompt (or after vcvars64.bat).
For every image and lambda: encode -> write the .sphota file -> decode from the file with a *second* codec instance ->
check torch.equal(decoded latent, transmitted latent), measure real bpp = 8 * file_size / (H * W) (header included),
the payload bpp without the header, the estimated bpp from the entropy model, PSNR / MS-SSIM / LPIPS of the decode,
and the encode + decode wall time. Outputs in --out: rd_real.csv, rd_real_points.csv, summary.json, files/*.sphota,
and report/figures/rd_kodak_real.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from eval_codec import baseline_curves  # noqa: E402  (scripts/eval_codec.py)
from truesight.codec import bitstream as bs  # noqa: E402
from truesight.codec.entropy_gmm import L, torchac_available  # noqa: E402
from truesight.codec.eval_codec import bd_table, beats_baseline, mean_point  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.codec.vae_codec import bpp_terms  # noqa: E402
from truesight.data import FullImageDataset  # noqa: E402
from truesight.metrics import lpips_dist, msssim, psnr  # noqa: E402

PROC = ROOT / "data" / "processed"
FIGS = ROOT / "report" / "figures"


def run_model(ckpt, mode, files, out_files, device, limit, save_all):
    net, cfg = load_codec(ckpt, device)
    dec_net, _ = load_codec(ckpt, device)                   # an independent decoder, as a receiver would have
    net.prepare_for_coding(); dec_net.prepare_for_coding()  # entropy_bottleneck.update(force=True) + CPU float32 h_s
    ds = FullImageDataset(files, multiple=net.DOWN)
    bs.encode_image(net, torch.rand(1, 3, 64, 64, device=device))         # warm-up: torchac import / first-call cost
    rows = []
    for i in range(len(ds) if limit is None else min(limit, len(ds))):
        xpad, (h, w), path = ds[i]
        x = xpad[None, :, :h, :w].to(device)
        t0 = time.perf_counter()
        blob, d = bs.encode_image(net, x, mode=mode, seed=i)
        t1 = time.perf_counter()
        name = f"{Path(path).stem}_lmbda{cfg['lmbda']:.4f}.sphota"
        if save_all or i == 0:
            (out_files / name).write_bytes(blob)
        x_hat, out, s = bs.decode_image(dec_net, blob)
        t2 = time.perf_counter()
        exact = bool(torch.equal(out["y_hat"], d["y_hat"]))
        exact_vs_encoder = bool(torch.equal(out["y_hat"], torch.round(net.g_a(bs.pad_to_multiple(x, net.DOWN)).cpu()).clamp(-L, L)))
        with torch.no_grad():
            lik = net(bs.pad_to_multiple(x, net.DOWN))["likelihoods"]
        est = sum(bpp_terms(lik, h * w).values()).item()
        # The training estimate floors every symbol probability at 1e-9 (30 bits); the arithmetic coder works with 16-bit
        # probabilities, so it can never charge more than ~16 bits. For a well-trained model almost no symbol is that
        # unlikely and the two agree; for a barely trained one the estimate over-charges. Report both.
        est16 = (-torch.log2(lik["y"].clamp(min=2.0 ** -16)).sum() - torch.log2(lik["z"]).sum()).item() / (h * w)
        frac_floored = float((lik["y"] < 2.0 ** -16).float().mean())
        payload = 8 * (len(s.z) + len(s.y)) / (h * w)
        x_hat = x_hat.cpu().clamp(0, 1)
        xo = x.cpu()
        rows.append(dict(lmbda=cfg["lmbda"], image=Path(path).name, file_bytes=len(blob), bpp=8 * len(blob) / (h * w),
                         bpp_payload=payload, bpp_est=est, payload_vs_est_pct=(payload / est - 1) * 100,
                         bpp_est_coder_floor=est16, payload_vs_est_coder_floor_pct=(payload / est16 - 1) * 100,
                         frac_y_below_coder_floor=frac_floored,
                         bit_exact=exact and exact_vs_encoder, n_clamped=d["n_clamped"], psnr=psnr(xo, x_hat),
                         msssim=msssim(xo, x_hat), lpips=lpips_dist(xo, x_hat), t_encode=t1 - t0, t_decode=t2 - t1))
        r = rows[-1]
        print(f"  lmbda {cfg['lmbda']:.4f} {r['image']}: {r['bpp']:.4f} bpp real vs {est:.4f} est "
              f"({r['payload_vs_est_pct']:+.2f}%), {r['psnr']:.2f} dB, exact={r['bit_exact']}, "
              f"enc {r['t_encode']:.1f}s dec {r['t_decode']:.1f}s", flush=True)
    return pd.DataFrame(rows)


def plot(points, base, path):
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
    g = points.sort_values("bpp")
    for a, col in zip(ax, ["psnr", "msssim", "lpips"]):
        a.plot(g.bpp, g[col], "k*-", ms=9, lw=2, label="TrueSight (real files)")
        a.plot(g.bpp_est, g[col], "x--", c="tab:orange", label="TrueSight (estimated bits)")
    for a, t in zip(ax, ["PSNR (dB)", "MS-SSIM", "LPIPS (lower = better)"]):
        a.set(xscale="log", xlabel="bits per pixel (real file size)", ylabel=t, title=f"{t.split(' ')[0]} vs bpp - Kodak"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=str(ROOT / "experiments" / "step7_codec"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step8_bitstream"))
    ap.add_argument("--test-list", default=str(PROC / "test_kodak.txt"))
    ap.add_argument("--figs", default=str(FIGS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-all", action="store_true")
    a = ap.parse_args()
    if not torchac_available():
        sys.exit("torchac is not importable (it compiles a C++ extension). On Windows run this from an 'x64 Native "
                 "Tools' prompt or after vcvars64.bat; otherwise install ninja and a C++ compiler.")
    out, figs = Path(a.out), Path(a.figs)
    (out / "files").mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(p for p in Path(a.exp).glob("lmbda*/final.pt"))             # K = 3 models only
    if not ckpts:
        sys.exit(f"no lmbda*/final.pt under {a.exp}; train with scripts/train_all_lambdas.py first")

    frames = []
    for mode, ck in enumerate(ckpts):
        frames.append(run_model(ck, mode, a.test_list, out / "files", a.device, a.limit, a.save_all))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "rd_real.csv", index=False)
    points = pd.DataFrame([mean_point(g.drop(columns=["lmbda", "bit_exact"]), lmbda=l,
                                      bit_exact=bool(g.bit_exact.all())) for l, g in df.groupby("lmbda")])
    points.to_csv(out / "rd_real_points.csv", index=False)

    base = baseline_curves()
    summary = {
        "images_per_model": int(df.groupby("lmbda").size().iloc[0]), "models": int(points.shape[0]),
        "all_bit_exact": bool(df.bit_exact.all()), "n_clamped_total": int(df.n_clamped.sum()),
        "payload_vs_estimate_pct": {"mean": float(df.payload_vs_est_pct.mean()),
                                    "max_abs": float(df.payload_vs_est_pct.abs().max())},
        "within_3pct_of_estimate": bool(df.payload_vs_est_pct.abs().max() < 3.0),
        "payload_vs_estimate_coder_floor_pct": {"mean": float(df.payload_vs_est_coder_floor_pct.mean()),
                                                "max_abs": float(df.payload_vs_est_coder_floor_pct.abs().max())},
        "within_3pct_of_coder_floor_estimate": bool(df.payload_vs_est_coder_floor_pct.abs().max() < 3.0),
        "mean_fraction_y_symbols_below_coder_floor": float(df.frac_y_below_coder_floor.mean()),
        "header_bytes": bs.header_bytes(0),
        "header_share_of_file_pct_mean": float((bs.header_bytes(0) / df.file_bytes).mean() * 100),
        "time_s": {"encode_mean": float(df.t_encode.mean()), "decode_mean": float(df.t_decode.mean()),
                   "encode_plus_decode_max": float((df.t_encode + df.t_decode).max())},
        "encode_decode_under_10s_on_cpu": bool((df.t_encode + df.t_decode).max() < 10.0) if a.device == "cpu" else None,
    }
    if len(points) >= 4:
        summary["bd_rate_psnr_pct"] = {k: round(v, 1) for k, v in bd_table(points, base, "psnr").items()}
        summary["bd_rate_msssim_pct"] = {k: round(v, 1) for k, v in bd_table(points, base, "msssim").items()}
    else:
        summary["note"] = "BD-rate needs >= 4 lambdas"
    if "JPEG" in base:
        ok, det = beats_baseline(points, base["JPEG"], 0.5)
        summary["beats_jpeg_below_0.5bpp"] = ok
    plot(points, base, figs / "rd_kodak_real.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: None if o != o else float(o)))
    print("\n" + points[["lmbda", "bpp", "bpp_payload", "bpp_est", "psnr", "msssim", "lpips", "bit_exact"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
