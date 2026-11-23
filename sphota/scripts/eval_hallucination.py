"""Step 13 evaluation: build the patch dataset, compare the 8 hallucination detectors, overlays, safe mode, encoder-side boxes.

    python scripts/eval_hallucination.py --codec-lo experiments/step7_codec/lmbda0.0018/final.pt \
        --diffusion experiments/step10_diffusion/final.pt --list data/processed/val.txt --limit 100
    python scripts/eval_hallucination.py ... --codec-hi experiments/step7_codec/lmbda0.0483/final.pt --encoder-boxes --tfr

For every image: encode with the lowest-lambda codec, decode with 3 seeds (plain diffusion by default; --guided adds the
Step 12 guidance, --with-roi the protected boxes), compute the six patch signals, the per-patch LPIPS against the original
and the label (LPIPS > 0.35 and positive detail gain = sharp but wrong). Then:
  * split BY IMAGE into train / test and compute the AUROC of all 8 detectors (target: the best >= 0.75),
  * learning curves (train vs validation AUROC against the training-set size) for logistic regression and XGBoost,
  * fit the decoder-side risk model (all signals) and the encoder-side one (coarse-decode signals only),
  * 10 uncertainty overlays, a contact sheet of 50 labelled patches to check the labels by eye,
  * safe mode: LPIPS / PSNR (and TFR with --tfr) of the decode before and after replacing flagged patches by the coarse decode,
  * --encoder-boxes: how many hallucinated patches fall inside the protected boxes with and without the risk boxes, and the bytes.
Outputs in --out: patches.csv, auroc.csv / auroc.json, summary.json, risk_full.joblib, risk_encoder.joblib; figures in --figs:
halluc_learning_curves.png, halluc_overlays.png, halluc_label_check.png. Needs torchac (real files): run from an x64 Native Tools
prompt on Windows.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from truesight import eval_limits  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from truesight.metrics import lpips_dist, psnr  # noqa: E402  (imports lpips before sklearn)
from truesight.analysis import hallucination as H  # noqa: E402
from truesight.analysis.risk_model import (RiskModel, best_detector, evaluate_detectors, learning_curve,  # noqa: E402
                                           split_by_image)
from sklearn.metrics import roc_auc_score  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.guidance import sphota_decode  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402
from truesight.protect import roi  # noqa: E402
from truesight.protect.detect import FaceDetector, TextDetector  # noqa: E402

PROC = ROOT / "data" / "processed"


def to_t(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


def label_sheet(samples, path, scale=3):
    """50 labelled patches: decode on top of the original, with LPIPS / detail gain / label, to check the labels by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    cols = 10
    rows = -(-n // cols)
    fig, ax = plt.subplots(rows, cols, figsize=(1.55 * cols, 2.1 * rows), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, s in zip(ax.ravel(), samples):
        both = np.concatenate([s["decode"], s["orig"]], axis=0)
        a.imshow(np.kron(both, np.ones((scale, scale, 1))).clip(0, 1), interpolation="nearest")
        a.set_title(f"{'HALLUC' if s['label'] else 'ok'}\nLP {s['lpips']:.2f} dG {s['gain']:.0f}", fontsize=6,
                    color="tab:red" if s["label"] else "k")
    fig.suptitle("Label check: top = decode patch, bottom = original patch", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def overlays_figure(items, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items)
    fig, ax = plt.subplots(n, 4, figsize=(13, 3.1 * n), squeeze=False)
    for r, it in enumerate(items):
        for c, (im, t) in enumerate(((it["orig"], "original"), (it["decode"], "decoded"), (it["overlay"], "uncertainty overlay"),
                                     (it["safe"], f"safe mode ({int(it['flags'].sum())} patches replaced)"))):
            ax[r][c].imshow(im); ax[r][c].axis("off")
            ax[r][c].set_title(t if r == 0 or c == 3 else "", fontsize=9)
        ax[r][0].set_ylabel(it["name"])
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def curves_figure(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.8), squeeze=False)
    for a, (name, lc) in zip(ax[0], curves.items()):
        n = [r["n_train"] for r in lc]
        a.plot(n, [r["train_auroc"] for r in lc], "o-", label="train AUROC")
        a.plot(n, [r["val_auroc"] for r in lc], "s-", label="validation AUROC")
        a.set(xlabel="training patches", ylabel="AUROC", title=name, xscale="log"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec-lo", required=True)
    ap.add_argument("--codec-hi", default=None, help="lambda 0.0483 codec (needed for --with-roi / --encoder-boxes)")
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--list", default=str(PROC / "val.txt"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=H.N_SEEDS)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--guided", action="store_true")
    ap.add_argument("--frozen", default=None, help="frozen.json of the guidance (with --guided)")
    ap.add_argument("--with-roi", action="store_true")
    ap.add_argument("--label-lpips", type=float, default=H.LABEL_LPIPS)
    ap.add_argument("--threshold", type=float, default=0.5, help="risk threshold for flagged patches / safe mode")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--encoder-boxes", action="store_true")
    ap.add_argument("--tfr", action="store_true")
    eval_limits.add_argument(ap)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step13_honesty"))
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out, figs = Path(a.out), Path(a.figs); out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    gk = {}
    if a.guided:
        fz = json.loads(Path(a.frozen).read_text())
        gk = dict(lr=fz["lr"], t_start=fz["t_start"], alpha=fz["alpha"], beta=fz["beta"], inner_steps=fz["inner_steps"], margin=fz["margin"])

    lo, _ = load_codec(a.codec_lo, dev); lo.prepare_for_coding()
    hi = lo
    if a.codec_hi:
        hi, _ = load_codec(a.codec_hi, dev); hi.prepare_for_coding()
    if (a.with_roi or a.encoder_boxes) and not a.codec_hi:
        sys.exit("--with-roi / --encoder-boxes need --codec-hi")
    unet, ucfg = load_unet(a.diffusion, dev); sched = Schedule(ucfg["T"], ucfg["kind"], device=dev)
    td, fd = TextDetector(), FaceDetector()
    files = read_list(a.list)[: a.limit]

    rows, per_image = [], []
    for idx, path in enumerate(files):
        pil, planned, text, faces, _ = eval_limits.limit_and_detect(Image.open(path).convert("RGB"), td, fd, a.max_pixels)
        x = to_t(pil).to(dev)
        detected = text + faces
        boxes = planned if a.with_roi else []
        blob, _ = roi.encode_protected(lo, hi, x, boxes)
        prot = roi.decode_protected(lo, hi, blob)
        decodes = [sphota_decode(lo, hi, unet, sched, prot=prot, w=a.w, steps=a.steps, seed=s, guided=a.guided,
                                 use_roi=a.with_roi, **gk).x for s in range(a.seeds)]
        t = H.patch_table(decodes, prot.coarse, prot.y_hat, lo, detected, prot.boxes if a.with_roi else (), original=x.cpu())
        labels = H.make_labels(t["lpips"], t["X"][:, H.FEATURES.index("detail_gain")], a.label_lpips)
        for p in range(len(labels)):
            rows.append(dict(image=idx, patch=p, label=int(labels[p]), lpips=t["lpips"][p],
                             **{f: t["X"][p, j] for j, f in enumerate(H.FEATURES)}))
        per_image.append(dict(name=Path(path).name, pil=pil, x=x.cpu(), decodes=decodes, coarse=prot.coarse.cpu(), prot=prot,
                              table=t, labels=labels, detected=detected, planned=planned, blob=blob))
        print(f"{Path(path).name}: {len(labels)} patches, {int(labels.sum())} hallucinated ({labels.mean():.0%})", flush=True)

    df = pd.DataFrame(rows); df.to_csv(out / "patches.csv", index=False)
    X, y, g = df[H.FEATURES].to_numpy(), df.label.to_numpy(), df.image.to_numpy()
    Z = H.transform_features(X)
    pos_rate = float(y.mean())
    print(f"\n{len(df)} patches from {len(files)} images, {pos_rate:.1%} hallucinated", flush=True)
    summary = {"images": len(files), "patches": int(len(df)), "positive_rate": pos_rate, "label_lpips": a.label_lpips,
               "positive_rate_in_5_20_pct": bool(0.05 <= pos_rate <= 0.20), "guided": a.guided, "with_roi": a.with_roi}
    if len(np.unique(y)) < 2:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        sys.exit("only one class among the labels: lower --label-lpips or use more / different images")
    tr, te = split_by_image(g, a.test_frac, seed=0)
    res = evaluate_detectors(Z[tr], y[tr], Z[te], y[te])
    name, best, met = best_detector(res)
    pd.Series(res, name="auroc").to_csv(out / "auroc.csv")
    summary.update(split={"train_images": int(len(set(g[tr]))), "test_images": int(len(set(g[te]))), "train_patches": int(len(tr)),
                          "test_patches": int(len(te)), "test_positive_rate": float(y[te].mean())},
                   auroc=res, best={"detector": name, "auroc": best, "target_0.75_met": met})
    print("AUROC (held-out images):\n" + pd.Series(res).round(4).to_string() + f"\nbest: {name} {best:.3f} (target 0.75: {met})", flush=True)

    curves = {m: learning_curve(m, Z[tr], y[tr], Z[te], y[te]) for m in ("logistic", "xgboost")}
    summary["learning_curves"] = curves
    curves_figure(curves, figs / "halluc_learning_curves.png")

    full = RiskModel(H.FEATURES).fit(X[tr], y[tr]); enc = RiskModel().fit(X[tr], y[tr])
    full.save(out / "risk_full.joblib"); enc.save(out / "risk_encoder.joblib")
    summary["risk_model_auroc_test"] = {"full": float(roc_auc_score(y[te], full.predict_risk(X[te]))),
                                        "encoder_only": float(roc_auc_score(y[te], enc.predict_risk(X[te])))}

    # 50 labelled patches to check by eye (25 hallucinated, 25 not, drawn at random)
    rng = np.random.default_rng(0)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    pick = np.concatenate([rng.choice(pos, min(25, len(pos)), replace=False), rng.choice(neg, min(25, len(neg)), replace=False)])
    samples = []
    for k in pick:
        im = per_image[int(df.image[k])]
        ph, pw = im["table"]["grid"]
        r_, c_ = divmod(int(df.patch[k]), pw)
        sl = (slice(None), slice(None), slice(r_ * H.PATCH, (r_ + 1) * H.PATCH), slice(c_ * H.PATCH, (c_ + 1) * H.PATCH))
        samples.append(dict(label=int(df.label[k]), lpips=float(df.lpips[k]), gain=float(df.detail_gain[k]),
                            orig=im["x"][sl][0].permute(1, 2, 0).numpy(), decode=im["decodes"][0].cpu()[sl][0].permute(1, 2, 0).numpy()))
    label_sheet(samples, figs / "halluc_label_check.png")

    # overlays and safe mode on the test images
    test_imgs = sorted(set(g[te]))
    shown, safe_rows = [], []
    for i in test_imgs:
        im = per_image[i]
        t = im["table"]
        risk = full.predict_risk(t["X"])
        grid = t["grid"]
        flags = (risk >= a.threshold).reshape(grid)
        Hh, Ww = im["x"].shape[-2:]
        decode = im["decodes"][0].cpu()
        prot_mask = im["prot"].mask.cpu() if a.with_roi and len(im["prot"].boxes) else None
        safe = H.safe_mode(decode, im["coarse"], flags, prot_mask)
        overlay = H.heat_overlay(decode, H.upsample_scores(risk, grid, Hh, Ww))
        row = dict(image=im["name"], flagged_fraction=float(flags.mean()),
                   lpips_decode=lpips_dist(im["x"], decode), lpips_safe=lpips_dist(im["x"], safe),
                   psnr_decode=psnr(im["x"], decode), psnr_safe=psnr(im["x"], safe))
        safe_rows.append(row)
        im["safe_pil"] = to_pil(safe)
        if len(shown) < 10:
            shown.append(dict(name=im["name"], orig=im["pil"], decode=to_pil(decode), overlay=to_pil(overlay), safe=im["safe_pil"], flags=flags))
    overlays_figure(shown, figs / "halluc_overlays.png")
    sr = pd.DataFrame(safe_rows)
    summary["safe_mode"] = {"test_images": len(sr), "mean_flagged_fraction": float(sr.flagged_fraction.mean()),
                            "lpips_decode": float(sr.lpips_decode.mean()), "lpips_safe": float(sr.lpips_safe.mean()),
                            "psnr_decode": float(sr.psnr_decode.mean()), "psnr_safe": float(sr.psnr_safe.mean())}
    if a.tfr:
        from truesight.protect.faithfulness import EasyOCRAdapter, text_faithfulness_rate
        ocr = EasyOCRAdapter()
        pairs_d = [(per_image[i]["pil"], to_pil(per_image[i]["decodes"][0].cpu())) for i in test_imgs]
        pairs_s = [(per_image[i]["pil"], per_image[i]["safe_pil"]) for i in test_imgs]
        summary["safe_mode"]["tfr_decode"] = text_faithfulness_rate(ocr, pairs_d)["tfr"]
        summary["safe_mode"]["tfr_safe"] = text_faithfulness_rate(ocr, pairs_s)["tfr"]

    # encoder-side protection: do the risk boxes cover the hallucinated patches?
    if a.encoder_boxes:
        eb = []
        for i in test_imgs:
            im = per_image[i]
            t = im["table"]
            Hh, Ww = im["x"].shape[-2:]
            ph, pw = t["grid"]
            Xenc = t["X"].copy()                         # the encoder sees the coarse decode and the detectors only
            risk = enc.predict_risk(Xenc)
            flags = (risk >= a.threshold).reshape(ph, pw)
            with_risk = H.add_risk_boxes(im["detected"], flags, Hh, Ww)
            base = im["planned"]
            hall = im["labels"].reshape(ph, pw).astype(bool)

            def covered(boxes):
                m = H.box_mask(Hh, Ww, boxes, xywh=True)
                cov = H.to_patches(m).numpy() > 0.5
                return float((cov & hall).sum() / max(1, hall.sum()))
            nb = {k: len(v) for k, v in (("base", base), ("with_risk", with_risk))}
            sizes = {}
            for k, bx in (("base", base), ("with_risk", with_risk)):
                blob, _ = roi.encode_protected(lo, hi, im["x"].to(dev), bx)
                sizes[k] = roi.size_breakdown(blob)
            eb.append(dict(image=im["name"], hallucinated=int(hall.sum()), coverage_base=covered(base), coverage_with_risk=covered(with_risk),
                           boxes_base=nb["base"], boxes_with_risk=nb["with_risk"], bpp_base=sizes["base"]["bpp_total"],
                           bpp_with_risk=sizes["with_risk"]["bpp_total"], roi_share_base=sizes["base"]["roi_share"],
                           roi_share_with_risk=sizes["with_risk"]["roi_share"]))
        e = pd.DataFrame(eb)
        e.to_csv(out / "encoder_boxes.csv", index=False)
        has = e[e.hallucinated > 0]
        summary["encoder_boxes"] = {"images": len(e), "images_with_hallucinations": len(has),
                                    "coverage_base": float(has.coverage_base.mean()) if len(has) else float("nan"),
                                    "coverage_with_risk": float(has.coverage_with_risk.mean()) if len(has) else float("nan"),
                                    "bpp_base": float(e.bpp_base.mean()), "bpp_with_risk": float(e.bpp_with_risk.mean())}

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k not in ("learning_curves",)}, indent=2, default=float))


if __name__ == "__main__":
    main()
