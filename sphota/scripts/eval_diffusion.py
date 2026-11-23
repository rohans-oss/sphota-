"""Step 10 evaluation: diffusion decode vs coarse decode at the same bits, CFG sweep, steps sweep, figures.

    python scripts/eval_diffusion.py                                   # full Kodak, guide defaults
    python scripts/eval_diffusion.py --limit 4 --steps 10 --device cpu # quick check
    python scripts/eval_diffusion.py --fid                             # also FID (needs `pip install clean-fid`; the
                                                                       # first use downloads Inception weights)

Uses the lowest-lambda codec (experiments/step7_codec/lmbda0.0018/final.pt) and the EMA diffusion checkpoint
(experiments/step10_diffusion/final.pt). Every method is compared at the SAME bpp: the coarse and diffusion decodes
come from the same latent (bpp = estimated bits of that latent / H*W); JPEG and WebP are encoded with the quality that
lands closest to that bpp (flagged 'reachable = False' when their size floor is above it, as it is at ~0.03 bpp).
Outputs in --out: sweep_cfg.csv, sweep_steps.csv, results.csv (per image), summary.json, tables.md; figures in --figs:
diff_cfg_sweep.png, diff_steps_sweep.png, diff_comparison.png, diff_cfg_grid.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from truesight.baselines.jpeg import find_quality_for_bpp  # noqa: E402
from truesight.codec.bitstream import pad_to_multiple  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.codec.vae_codec import bpp_terms  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.decode import coarse_decode, diffusion_decode  # noqa: E402
from truesight.diffusion.sample import FirstStepCache  # noqa: E402
from truesight.diffusion.eval_diffusion import checks, choose_default_w, fmt_table, summarize  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402
from truesight.metrics import lpips_dist, msssim, psnr  # noqa: E402

PROC = ROOT / "data" / "processed"
W_SWEEP = [0.0, 1.0, 1.5, 2.0, 3.0]
STEPS_SWEEP = [10, 25, 50]


def to_t(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def metrics(x, y):
    return dict(psnr=psnr(x, y), msssim=msssim(x, y), lpips=lpips_dist(x, y))


def prepare(files, codec, device, limit):
    """Per image: original, padded coarse decode (the condition), estimated bpp, size."""
    items = []
    for path in files[: limit or len(files)]:
        pil = Image.open(path).convert("RGB")
        x = to_t(pil).to(device)
        h, w = x.shape[-2:]
        xp = pad_to_multiple(x, codec.DOWN)
        with torch.no_grad():
            lik = codec(xp)["likelihoods"]
            coarse = coarse_decode(codec, xp)
        items.append(dict(name=Path(path).name, pil=pil, x=x.cpu(), h=h, w=w, coarse_pad=coarse,
                          bpp=sum(bpp_terms(lik, h * w).values()).item()))
    return items


def run_decode(items, unet, sched, w, steps, seed, tile, device, cache):
    """Diffusion-decode every image (cached by (w, steps)). Returns per-image dicts incl. the decoded tensor.
    The first denoising step (same noise, same condition, t = T-1) is identical for every w and every step count: it is computed once
    per image and reused (``FirstStepCache`` per image; the check inside compares the actual tensors)."""
    key = (w, steps)
    if key not in cache:
        rows = []
        for it in items:
            t0 = time.perf_counter()
            fs = it.setdefault("first_step", FirstStepCache())
            out = diffusion_decode(unet, sched, it["coarse_pad"], steps=steps, w=w, seed=seed, tile=tile, first_step=fs if tile is None else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            y = out[..., : it["h"], : it["w"]].cpu()
            rows.append(dict(image=it["name"], w=w, steps=steps, time=dt, y=y, **metrics(it["x"], y)))
        cache[key] = rows
    return cache[key]


def sweep_frame(rows_by_key, key_name):
    recs = [dict({key_name: k}, **{m: float(np.mean([r[m] for r in rows])) for m in ("psnr", "msssim", "lpips", "time")})
            for k, rows in rows_by_key.items()]
    return pd.DataFrame(recs)


def save_fig_cfg(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(df.w, df.lpips, "o-"); ax[0].set(xlabel="CFG scale w", ylabel="mean LPIPS (lower = better)", title="LPIPS vs w")
    ax[1].plot(df.w, df.psnr, "s-", c="tab:red"); ax[1].set(xlabel="CFG scale w", ylabel="mean PSNR (dB)", title="PSNR vs w")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_fig_steps(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, col, t in zip(ax, ["lpips", "psnr", "time"], ["mean LPIPS", "mean PSNR (dB)", "decode time per image (s)"]):
        a.plot(df.steps, df[col], "o-"); a.set(xlabel="DDIM steps", ylabel=t, title=t); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def crop_center(t, size):
    h, w = t.shape[-2:]
    y0, x0 = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return t[0, :, y0:y0 + size, x0:x0 + size].permute(1, 2, 0).clamp(0, 1).numpy()


def save_fig_comparison(items, main_rows, base, path, n=3, size=192):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n, len(items))
    cols = ["original", "coarse", "diffusion", "JPEG", "WebP"]
    fig, ax = plt.subplots(n, 5, figsize=(3.2 * 5, 3.2 * n), squeeze=False)
    for r in range(n):
        it, row = items[r], main_rows[r]
        imgs = [it["x"], it["coarse_pad"][..., : it["h"], : it["w"]].cpu(), row["y"],
                to_t(base["jpeg"][r]["rec"]), to_t(base["webp"][r]["rec"])]
        for c, (im, name) in enumerate(zip(imgs, cols)):
            ax[r][c].imshow(crop_center(im, size)); ax[r][c].axis("off")
            if r == 0:
                ax[r][c].set_title(name)
        ax[r][0].set_ylabel(it["name"])
    fig.suptitle(f"Same bits ({np.mean([i['bpp'] for i in items[:n]]):.3f} bpp est.); JPEG/WebP at their closest quality", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_fig_cfg_grid(items, cache, ws, steps, path, size=160):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = items[0]
    fig, ax = plt.subplots(1, len(ws) + 2, figsize=(2.6 * (len(ws) + 2), 2.8))
    ax[0].imshow(crop_center(it["x"], size)); ax[0].set_title("original", fontsize=9)
    ax[1].imshow(crop_center(it["coarse_pad"].cpu(), size)); ax[1].set_title("coarse", fontsize=9)
    for a, w in zip(ax[2:], ws):
        a.imshow(crop_center(cache[(w, steps)][0]["y"], size)); a.set_title(f"w = {w:g}", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", default=str(ROOT / "experiments" / "step7_codec" / "lmbda0.0018" / "final.pt"))
    ap.add_argument("--diffusion", default=str(ROOT / "experiments" / "step10_diffusion" / "final.pt"))
    ap.add_argument("--test-list", default=str(PROC / "test_kodak.txt"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step10_diffusion" / "eval"))
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--steps", type=int, default=25, help="DDIM steps for the CFG sweep")
    ap.add_argument("--main-steps", type=int, default=50, help="DDIM steps for the headline table")
    ap.add_argument("--steps-sweep", type=int, nargs="*", default=STEPS_SWEEP, help="step counts of the steps sweep")
    ap.add_argument("--w", type=float, default=None, help="CFG scale for the headline table (default: chosen by the sweep)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tile", type=int, default=None, help="decode in tiles of this size (multiple of 8) if memory is short")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fid", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device)
    out, figs = Path(a.out), Path(a.figs); out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)

    codec, ccfg = load_codec(a.codec, device)
    unet, ucfg = load_unet(a.diffusion, device, use_ema=True)
    sched = Schedule(ucfg["T"], ucfg["kind"], device=device)
    items = prepare(read_list(a.test_list), codec, device, a.limit)
    print(f"{len(items)} images, codec lambda {ccfg['lmbda']}, mean est. bpp {np.mean([i['bpp'] for i in items]):.4f}", flush=True)
    cache = {}

    # ---- CFG sweep (fixed steps) -> default w
    cfg_rows = {w: run_decode(items, unet, sched, w, a.steps, a.seed, a.tile, device, cache) for w in W_SWEEP}
    cfg_df = sweep_frame(cfg_rows, "w"); cfg_df.to_csv(out / "sweep_cfg.csv", index=False)
    w_def, reason = choose_default_w(cfg_df)
    w_main = a.w if a.w is not None else w_def
    print(f"CFG sweep:\n{cfg_df.to_string(index=False)}\ndefault w = {w_def}: {reason}", flush=True)

    # ---- steps sweep at the chosen w
    steps_rows = {s: run_decode(items, unet, sched, w_main, s, a.seed, a.tile, device, cache)
                  for s in sorted(set(list(a.steps_sweep) + [a.main_steps]))}
    steps_df = sweep_frame(steps_rows, "steps"); steps_df.to_csv(out / "sweep_steps.csv", index=False)
    print(f"steps sweep (w={w_main}):\n{steps_df.to_string(index=False)}", flush=True)

    # ---- headline table: coarse vs diffusion vs JPEG / WebP at the same bpp
    main_rows = steps_rows[a.main_steps]
    base = {"jpeg": [], "webp": []}
    long = []
    for it, row in zip(items, main_rows):
        long.append(dict(image=it["name"], method="coarse", bpp=it["bpp"], time=0.0,
                         **metrics(it["x"], it["coarse_pad"][..., : it["h"], : it["w"]].cpu())))
        long.append(dict(image=it["name"], method="diffusion", bpp=it["bpp"], time=row["time"],
                         **{k: row[k] for k in ("psnr", "msssim", "lpips")}))
        for fmt in ("jpeg", "webp"):
            q, b, rec, reachable = find_quality_for_bpp(it["pil"], it["bpp"], fmt.upper())
            base[fmt].append(dict(rec=rec))
            long.append(dict(image=it["name"], method=fmt, bpp=b, time=0.0, quality=q, reachable=reachable,
                             **metrics(it["x"], to_t(rec))))
    per_image = pd.DataFrame(long); per_image.to_csv(out / "results.csv", index=False)
    summary = summarize(per_image)
    result = {"images": len(items), "codec_lmbda": ccfg["lmbda"], "w_main": w_main, "main_steps": a.main_steps,
              "default_w": w_def, "default_w_reason": reason, "table": summary.round(4).reset_index().to_dict("records"),
              "decode_time_per_image_s": {int(s): float(np.mean([r["time"] for r in rows])) for s, rows in steps_rows.items()},
              "jpeg_webp_reachable_fraction": {f: float(per_image[per_image.method == f].reachable.mean()) for f in ("jpeg", "webp")}}

    # ---- optional FID
    fid = None
    if a.fid:
        try:
            from cleanfid import fid as cfid
        except ImportError:
            sys.exit("--fid needs clean-fid: pip install clean-fid (it downloads Inception weights on first use)")
        dirs = {m: out / "fid" / m for m in ("original", "coarse", "diffusion", "jpeg", "webp")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for i, (it, row) in enumerate(zip(items, main_rows)):
            for m, im in (("original", it["pil"]), ("coarse", Image.fromarray((crop_to(it["coarse_pad"], it) * 255).round().astype("uint8"))),
                          ("diffusion", Image.fromarray((row["y"][0].permute(1, 2, 0).numpy() * 255).round().astype("uint8"))),
                          ("jpeg", base["jpeg"][i]["rec"]), ("webp", base["webp"][i]["rec"])):
                im.save(dirs[m] / f"{i:03d}.png")
        fid = {m: float(cfid.compute_fid(str(dirs["original"]), str(dirs[m]), mode="clean", device=device))
               for m in ("diffusion", "coarse", "jpeg", "webp")}
        result["fid"] = fid
        result["fid_note"] = f"computed on only {len(items)} images: indicative, not a reliable absolute FID"
    result["checks"] = checks(summary, fid)

    (out / "summary.json").write_text(json.dumps(result, indent=2, default=float))
    (out / "tables.md").write_text("## Coarse vs diffusion at the same bpp\n\n" + fmt_table(summary) +
                                   "\n\n## CFG sweep\n\n" + fmt_table(cfg_df.set_index("w")) +
                                   "\n\n## Steps sweep\n\n" + fmt_table(steps_df.set_index("steps")) + "\n")
    save_fig_cfg(cfg_df, figs / "diff_cfg_sweep.png"); save_fig_steps(steps_df, figs / "diff_steps_sweep.png")
    save_fig_comparison(items, main_rows, base, figs / "diff_comparison.png")
    save_fig_cfg_grid(items, cache, W_SWEEP, a.steps, figs / "diff_cfg_grid.png")
    print("\n" + (out / "tables.md").read_text() + "\n" + json.dumps(result["checks"], indent=2))


def crop_to(t, it):
    return t[0, :, : it["h"], : it["w"]].permute(1, 2, 0).cpu().numpy()


if __name__ == "__main__":
    main()
