"""Does fp16 mixed precision change the codec's rate-distortion quality? A short A/B run that decides it with numbers.

    python scripts/amp_parity.py                                   # lambda 0.0130, 3000 its, 2 seeds per arm (about an hour on a T4)
    python scripts/amp_parity.py --iters 10000 --seeds 0 1 2       # stronger evidence (about 4-5 hours)
    TS_AMP_PARITY_FILE=experiments/amp_parity.json TS_CODEC_AMP=auto python -m truesight.codec.train_codec ...   # use only if it passed

Arm "fp32": the guide's training (pure fp32). Arm "amp": fp16 for the plain convolutions, fp32 for GDN, the last layer of every
stack (so y, z, mu, sigma and the reconstruction keep fp32 precision), the entropy model, the likelihoods, the rate sum, the MSE and
the auxiliary loss (TrueSightCodec._seq). Both arms use the same seeds, data, batch, learning rate and iteration count. Each trained
codec is then evaluated in plain fp32 with real rounding on the same held-out images (centre crops), and the arms are compared:

    PASS  when  |mean PSNR difference| <= max(0.05 dB, 2 x the PSNR spread between the fp32 seeds)
          and   |relative bpp difference| <= max(0.5 %, 2 x the bpp spread between the fp32 seeds)
          and   fewer than 1 % of the amp steps were skipped by the loss scaler, and nothing was non-finite.
The tolerances follow the research report (seed spread, roughly 0.02-0.05 dB and 0.2-0.5 % bpp). The result, with the measured speed-up
(seconds per iteration, fp32 / amp), is written to <out>/amp_parity.json; ``amp_ok`` is what TS_CODEC_AMP=auto reads. Without a CUDA GPU
there is no fp16 path: the amp arm is then identical to fp32 and the script says so.

A 3000-iteration run says whether fp16 *starts* the same; for the final decision the report also suggests a half-length (75k) run at one
lambda: ``python -m truesight.codec.train_codec --lmbda 0.0130 --iters 75000 --amp 1 ...`` against the fp32 run, compared with eval_codec.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from truesight import runtime  # noqa: E402
from truesight.codec.train_codec import load_codec, train_codec  # noqa: E402
from truesight.codec.vae_codec import bpp_terms  # noqa: E402
from truesight.data import read_list  # noqa: E402
from PIL import Image  # noqa: E402


def center_crop_batch(path, size):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(size, (min(w, h) // 64) * 64)
    x0, y0 = (w - s) // 2, (h - s) // 2
    a = np.asarray(im.crop((x0, y0, x0 + s, y0 + s)), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


@torch.no_grad()
def evaluate(final, files, size, device):
    """Mean (bpp, PSNR) of a trained codec on centre crops, in fp32 with real rounding (eval mode)."""
    net, _ = load_codec(final, device)
    net.amp = False
    bpps, psnrs = [], []
    for f in files:
        x = center_crop_batch(f, size).to(device)
        out = net(x)
        bpps.append(sum(bpp_terms(out["likelihoods"], x.shape[-1] * x.shape[-2]).values()).item())
        mse = torch.mean((out["x_hat"].clamp(0, 1) - x) ** 2).item()
        psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(bpps)), float(np.mean(psnrs))


def verdict(fp32, amp, speed, iters, skipped, finite=True):
    """fp32 / amp: lists of (bpp, psnr), one per seed. Returns the decision dict (pure function, unit tested)."""
    b32, p32 = np.array([r[0] for r in fp32]), np.array([r[1] for r in fp32])
    ba, pa = np.array([r[0] for r in amp]), np.array([r[1] for r in amp])
    dpsnr = float(pa.mean() - p32.mean())
    dbpp = float((ba.mean() - b32.mean()) / b32.mean())
    sd_p = float(p32.std(ddof=1)) if len(p32) > 1 else 0.0
    sd_b = float((b32 / b32.mean()).std(ddof=1)) if len(b32) > 1 else 0.0
    tol_p, tol_b = max(0.05, 2 * sd_p), max(0.005, 2 * sd_b)
    skip_frac = skipped / max(1, iters * len(amp))
    ok = bool(finite and abs(dpsnr) <= tol_p and abs(dbpp) <= tol_b and skip_frac < 0.01)
    return {"amp_ok": ok, "delta_psnr_db": dpsnr, "delta_bpp_rel": dbpp, "psnr_tolerance_db": tol_p, "bpp_tolerance_rel": tol_b,
            "fp32_psnr_seed_std": sd_p, "fp32_bpp_seed_rel_std": sd_b, "skipped_step_fraction": skip_frac, "speedup": speed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmbda", type=float, default=0.0130)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--M", type=int, default=192)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--val-list", default=str(ROOT / "data" / "processed" / "val.txt"))
    ap.add_argument("--eval-images", type=int, default=12)
    ap.add_argument("--eval-size", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "amp_parity"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = read_list(a.val_list)[: a.eval_images]
    if device.type != "cuda":
        print("NOTE: no CUDA GPU, so there is no fp16 path: the amp arm is identical to fp32 and this result says nothing.", flush=True)
    res = {"fp32": [], "amp": []}
    secs = {"fp32": 0.0, "amp": 0.0}
    skipped = 0
    for seed in a.seeds:
        for arm in ("fp32", "amp"):
            d = out / f"{arm}_seed{seed}"
            if not (d / "final.pt").exists():
                t0 = time.perf_counter()
                net = train_codec(a.lmbda, d, N=a.N, M=a.M, iters=a.iters, batch=a.batch, crop=a.crop, workers=a.workers, seed=seed,
                                  train_list=a.train_list, ckpt_every=10 ** 9, amp=(arm == "amp"), device=device)
                dt = time.perf_counter() - t0
                (d / "timing.json").write_text(json.dumps({"seconds": dt, "iters": a.iters, "skipped": getattr(net, "_amp_skipped", 0)}))
            tm = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {"seconds": 0, "skipped": 0}
            secs[arm] += tm["seconds"]
            if arm == "amp":
                skipped += tm.get("skipped", 0)
            res[arm].append(evaluate(d / "final.pt", files, a.eval_size, device))
            print(f"seed {seed} {arm}: bpp {res[arm][-1][0]:.4f} PSNR {res[arm][-1][1]:.3f} dB", flush=True)
    speed = (secs["fp32"] / secs["amp"]) if secs["amp"] > 0 else None
    v = verdict(res["fp32"], res["amp"], speed, a.iters, skipped, finite=all(np.isfinite(r).all() for arm in res.values() for r in arm))
    v.update(lmbda=a.lmbda, iters=a.iters, seeds=a.seeds, eval_images=len(files), device=str(device), results=res)
    (out / "amp_parity.json").write_text(json.dumps(v, indent=2))
    (out.parent / "amp_parity.json").write_text(json.dumps(v, indent=2))        # where TS_AMP_PARITY_FILE points by default
    print(json.dumps({k: v[k] for k in v if k != "results"}, indent=2))
    print("PASS: fp16 hybrid matches fp32 within the seed spread." if v["amp_ok"] else
          "NOT PROVEN: keep pure fp32 (or run a longer comparison). TS_CODEC_AMP=auto will stay off.")


if __name__ == "__main__":
    runtime.run_cli(main)
