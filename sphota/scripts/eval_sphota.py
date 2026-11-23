"""Step 12 evaluation: the main results table, ablations A4 / A5, L_bin plot and before/after examples.

    python scripts/eval_sphota.py --set text --limit 100 --codec-lo ... --codec-hi ... --diffusion ... \
        --frozen experiments/step12_guidance/frozen.json --tfr
    python scripts/eval_sphota.py --set face --limit 100 ... --fir

Methods, all decoded from the SAME .sphota file (same bits; the ROI streams are part of every bpp reported):
  coarse       the VAE codec alone (no generative step)
  plain        plain diffusion decode: no guidance, no protected regions
  roi_only     diffusion + feathered paste of the protected regions                    (ablation A5 on, A4 off)
  guided_only  diffusion + bin-consistency guidance, no protected regions              (A4 on, A5 off)
  sphota       diffusion + bin-consistency guidance + ROI-lock + feathered paste       (the full system)
Metrics: consistency rate (fraction of latent values of g_a(decode) inside the transmitted bin), LPIPS, PSNR, and with
--tfr / --fir the Text Faithfulness Rate / Face Identity Rate. Writes into --out: table.csv, summary.json (incl. the
done-criteria check), tables.md; figures: guidance_lbin.png (L_bin along the denoising steps) and guidance_examples.png
(plain diffusion vs Sphoṭa where plain changed the text / a face and Sphoṭa did not).
Needs torchac (real .sphota files): on Windows run from an x64 Native Tools prompt.
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
from PIL import Image  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.eval_guidance import (METHODS, done_criteria, mean_history, wins_by_cosine,  # noqa: E402
                                                wins_by_pass)
from truesight.diffusion.eval_diffusion import fmt_table  # noqa: E402
from truesight.diffusion.guidance import consistency_rate, sphota_decode  # noqa: E402
from truesight.diffusion.sample import PrefixCache  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402
from truesight.metrics import lpips_dist, psnr  # noqa: E402
from truesight.protect import roi  # noqa: E402
from truesight import eval_limits  # noqa: E402
from truesight.protect.detect import FaceDetector, TextDetector  # noqa: E402

PROC = ROOT / "data" / "processed"
SETS = {"text": "test_text", "face": "test_face", "kodak": "test_kodak", "clic": "test_clic"}
VARIANTS = {"plain": dict(guided=False, use_roi=False), "roi_only": dict(guided=False, use_roi=True),
            "guided_only": dict(guided=True, use_roi=False), "sphota": dict(guided=True, use_roi=True)}


def to_t(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


def plot_lbin(hist_plain, hist_guided, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4))
    if len(hist_plain):
        ax.plot(hist_plain[:, 0], hist_plain[:, 1], "o-", c="tab:red", ms=3, label="unguided: L_bin of x0_hat")
    if len(hist_guided):
        ax.plot(hist_guided[:, 0], hist_guided[:, 1], "s--", c="tab:gray", ms=3, label="guided: before the correction")
        ax.plot(hist_guided[:, 0], hist_guided[:, 3], "o-", c="tab:green", ms=3, label="guided: after the correction")
    ax.set(xlabel="DDIM timestep t (noisy -> clean)", ylabel="mean L_bin (0 = inside every transmitted bin)",
           title="Bin-consistency loss along the denoising steps", yscale="log")
    ax.invert_xaxis(); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def crop_around(pil, box, pad=24):
    x0, y0, x1, y1 = box
    return pil.crop((max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(pil.width, int(x1 + pad)), min(pil.height, int(y1 + pad))))


def examples_figure(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(examples)
    fig, ax = plt.subplots(n, 3, figsize=(10, 2.6 * n), squeeze=False)
    for r, e in enumerate(examples):
        for c, (im, t) in enumerate(((e["orig"], "original"), (e["plain"], "plain diffusion"), (e["sphota"], "Sphoṭa"))):
            ax[r][c].imshow(crop_around(im, e["box"])); ax[r][c].axis("off")
            ax[r][c].set_title(t if r == 0 else "", fontsize=9)
        ax[r][0].text(0, -0.08, e["caption"][0], transform=ax[r][0].transAxes, fontsize=7, va="top")
        ax[r][1].text(0, -0.08, e["caption"][1], transform=ax[r][1].transAxes, fontsize=7, va="top", color="tab:red")
        ax[r][2].text(0, -0.08, e["caption"][2], transform=ax[r][2].transAxes, fontsize=7, va="top", color="tab:green")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), default="text")
    ap.add_argument("--list", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--codec-lo", required=True)
    ap.add_argument("--codec-hi", required=True)
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--frozen", default=str(ROOT / "experiments" / "step12_guidance" / "frozen.json"))
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tfr", action="store_true")
    ap.add_argument("--fir", action="store_true")
    eval_limits.add_argument(ap)
    ap.add_argument("--hist-images", type=int, default=4, help="images used for the L_bin figure")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step12_guidance" / "eval"))
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out, figs = Path(a.out), Path(a.figs); out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    fz = json.loads(Path(a.frozen).read_text())                       # the values frozen by tune_guidance.py
    gk = dict(lr=fz["lr"], t_start=fz["t_start"], alpha=fz["alpha"], beta=fz["beta"], inner_steps=fz["inner_steps"], margin=fz["margin"])
    w, steps = a.w if a.w is not None else fz["w"], a.steps if a.steps is not None else fz["steps"]
    print(f"frozen guidance: {gk}, w={w}, steps={steps}", flush=True)

    lo, _ = load_codec(a.codec_lo, dev); hi, _ = load_codec(a.codec_hi, dev)
    lo.prepare_for_coding(); hi.prepare_for_coding()
    unet, ucfg = load_unet(a.diffusion, dev); sched = Schedule(ucfg["T"], ucfg["kind"], device=dev)
    td, fd = TextDetector(), FaceDetector()
    ocr = emb = None
    if a.tfr:
        from truesight.protect.faithfulness import EasyOCRAdapter
        ocr = EasyOCRAdapter()
    if a.fir:
        from truesight.protect.faithfulness import FaceEmbedder
        emb = FaceEmbedder()

    files = read_list(a.list or PROC / f"{SETS[a.set]}.txt")[: a.limit]
    originals, decodes, rows = [], {m: [] for m in METHODS}, []
    hist_plain, hist_guided = [], []
    for i, path in enumerate(files):
        pil, boxes, _, _, _ = eval_limits.limit_and_detect(Image.open(path).convert("RGB"), td, fd, a.max_pixels)
        x = to_t(pil).to(dev)
        blob, _ = roi.encode_protected(lo, hi, x, boxes)
        prot = roi.decode_protected(lo, hi, blob)
        br = roi.size_breakdown(blob)
        originals.append(pil)
        outs = {"coarse": prot.coarse.clamp(0, 1)}
        prefix = PrefixCache()          # plain / ROI / guided decodes of this file share the steps before their guidance starts
        for name, kw in VARIANTS.items():
            track = (name == "plain" and i < a.hist_images)
            o = sphota_decode(lo, hi, unet, sched, prot=prot, w=w, steps=steps, track_only=track, prefix=prefix, **kw, **gk)
            outs[name] = o.x
            if track:
                hist_plain.append(o.guide.history)
            if name == "sphota" and i < a.hist_images and o.guide is not None:
                hist_guided.append(o.guide.history)
        for m, y in outs.items():
            decodes[m].append(to_pil(y))
            rows.append(dict(image=Path(path).name, method=m, consistency=consistency_rate(lo, y, prot.y_hat),
                             lpips=lpips_dist(x.cpu(), y.cpu()), psnr=psnr(x.cpu(), y.cpu()), bpp=br["bpp_total"],
                             roi_share=br["roi_share"], n_boxes=br["n_boxes"]))
        print(f"{Path(path).name}: {len(boxes)} boxes, {br['bpp_total']:.3f} bpp, ROI {br['roi_share']:.0%}, consistency "
              + ", ".join(f"{r['method']} {r['consistency']:.2f}" for r in rows[-5:]), flush=True)

    df = pd.DataFrame(rows); df.to_csv(out / "per_image.csv", index=False)
    table = df.groupby("method")[["consistency", "lpips", "psnr", "bpp", "roi_share"]].mean().reindex(list(METHODS))
    wins, tfr_res, fir_res = [], {}, {}
    if ocr is not None:
        from truesight.protect.faithfulness import text_faithfulness_rate
        tfr_res = {m: text_faithfulness_rate(ocr, zip(originals, decodes[m])) for m in METHODS}
        table["tfr"] = pd.Series({m: tfr_res[m]["tfr"] for m in METHODS})
        pp = [(r["passed"] if r else None) for r in tfr_res["plain"]["per_image"]]
        sp = [(r["passed"] if r else None) for r in tfr_res["sphota"]["per_image"]]
        for i in wins_by_pass(pp, sp):
            bad = [g for g in tfr_res["plain"]["per_image"][i]["regions"] if g["cer"] > 0.05][0]
            good = next(g for g in tfr_res["sphota"]["per_image"][i]["regions"] if g["box"] == bad["box"])
            wins.append(dict(kind="text", image=files[i], box=bad["box"], orig=originals[i], plain=decodes["plain"][i],
                             sphota=decodes["sphota"][i],
                             caption=(f"OCR of the original: '{bad['original']}'", f"plain reads: '{bad['decoded']}' (CER {bad['cer']:.2f})",
                                      f"Sphoṭa reads: '{good['decoded']}' (CER {good['cer']:.2f})")))
    if emb is not None:
        from truesight.protect.faithfulness import face_identity_rate
        fir_res = {m: face_identity_rate(fd, emb, zip(originals, decodes[m])) for m in METHODS}
        table["fir"] = pd.Series({m: fir_res[m]["fir"] for m in METHODS})
        pf = [(i, f) for i, res in enumerate(fir_res["plain"]["per_image"]) for f in res]
        sf = [(i, f) for i, res in enumerate(fir_res["sphota"]["per_image"]) for f in res]
        for k in wins_by_cosine([f["cosine"] for _, f in pf], [f["cosine"] for _, f in sf]):
            i = pf[k][0]
            wins.append(dict(kind="face", image=files[i], box=pf[k][1]["box"], orig=originals[i], plain=decodes["plain"][i],
                             sphota=decodes["sphota"][i],
                             caption=("original face", f"plain: identity cosine {pf[k][1]['cosine']:.2f} (changed)",
                                      f"Sphoṭa: identity cosine {sf[k][1]['cosine']:.2f} (kept)")))
    crit = done_criteria({m: {k: float(v) for k, v in table.loc[m].items() if not pd.isna(v)} for m in table.index},
                         n_examples=len(wins) if (ocr is not None or emb is not None) else None)
    summary = {"set": a.set, "images": len(files), "frozen": gk, "w": w, "steps": steps, "table": table.round(4).reset_index().to_dict("records"),
               "criteria": crit, "n_examples": len(wins),
               "tfr_counts": {m: {k: v for k, v in r.items() if k != "per_image"} for m, r in tfr_res.items()},
               "fir_counts": {m: {k: v for k, v in r.items() if k != "per_image"} for m, r in fir_res.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    table.to_csv(out / "table.csv")
    (out / "tables.md").write_text(fmt_table(table) + "\n")
    plot_lbin(mean_history(hist_plain), mean_history(hist_guided), figs / "guidance_lbin.png")
    if wins:
        examples_figure(wins[:8], figs / "guidance_examples.png")
    print("\n" + table.round(4).to_string() + "\n\n" + json.dumps(crit, indent=2, default=float))


if __name__ == "__main__":
    main()
