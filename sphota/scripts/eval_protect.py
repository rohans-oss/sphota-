"""Step 11 evaluation: protected boxes, ROI cost, feathered seams, TFR / FIR.

    python scripts/eval_protect.py --set face --limit 20 --draw             # detect + draw boxes only (no codecs needed)
    python scripts/eval_protect.py --set text --codec-lo ... --codec-hi ... # + ROI file sizes, ROI share of the bytes
    python scripts/eval_protect.py --set text --codec-lo ... --codec-hi ... --diffusion ... --tfr --fir
                                                                            # + plain diffusion vs diffusion + ROI paste

--draw        save every image with its boxes drawn (red = text, blue = faces, green = final aligned boxes) to --figs/boxes
              so they can be checked by eye, plus 'seams_*.png' crops around the box edges of the pasted result.
--codec-lo / --codec-hi   final.pt of the lowest-lambda (base layer) and the highest-lambda (0.0483, protected regions)
              codecs; writes real .sphota files with the boxes (needs torchac: run from an x64 Native Tools prompt on Windows).
--diffusion   final.pt of the diffusion decoder: reports plain diffusion and diffusion + feathered ROI paste.
--tfr / --fir compute the Text Faithfulness Rate / Face Identity Rate for every method. These load the EasyOCR
              recognition model and the VGGFace2 face embedder, whose weights are downloaded on first use.
Outputs in --out: boxes.csv (per image), summary.json.
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
from PIL import Image, ImageDraw  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.protect import roi  # noqa: E402
from truesight.protect.detect import FaceDetector, TextDetector  # noqa: E402

PROC = ROOT / "data" / "processed"
SETS = {"text": "test_text", "face": "test_face", "kodak": "test_kodak", "clic": "test_clic"}


def to_t(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


def draw_boxes(pil, text, faces, final):
    im = pil.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for b in text:
        d.rectangle(b, outline=(255, 0, 0), width=2)
    for b in faces:
        d.rectangle(b, outline=(0, 90, 255), width=2)
    for x, y, w, h in final:
        d.rectangle((x, y, x + w - 1, y + h - 1), outline=(0, 200, 0), width=3)
    return im


def seam_figure(images, boxes, path, size=96):
    """Crops (original | pasted) around the top-left corner of up to 3 boxes: seams must be invisible at 1:1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(n, a, b) for n, a, b, bx in images if bx][:3]
    if not rows:
        return
    fig, ax = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)), squeeze=False)
    for r, (name, plain, pasted) in enumerate(rows):
        x, y, w, h = boxes[name][0]
        x0, y0 = max(0, x - size // 2), max(0, y - size // 2)
        for c, (im, t) in enumerate(((plain, "without paste"), (pasted, "with feathered ROI paste"))):
            ax[r][c].imshow(im.crop((x0, y0, x0 + size, y0 + size)).resize((size * 3, size * 3), Image.NEAREST)); ax[r][c].axis("off")
            ax[r][c].set_title(t if r == 0 else "", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), default="text")
    ap.add_argument("--list", default=None, help="explicit split list (overrides --set)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--codec-lo", default=None)
    ap.add_argument("--codec-hi", default=None)
    ap.add_argument("--diffusion", default=None)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--tfr", action="store_true")
    ap.add_argument("--fir", action="store_true")
    eval_limits.add_argument(ap)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step11_protect"))
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out, figs = Path(a.out), Path(a.figs)
    (figs / "boxes").mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    files = read_list(a.list or PROC / f"{SETS[a.set]}.txt")[: a.limit]

    text_det, face_det = TextDetector(), FaceDetector()
    use_codec = bool(a.codec_lo and a.codec_hi)
    if use_codec:
        lo, _ = load_codec(a.codec_lo, a.device); hi, _ = load_codec(a.codec_hi, a.device)
        lo.prepare_for_coding(); hi.prepare_for_coding()
    unet = sched = None
    if a.diffusion:
        from truesight.diffusion.decode import diffusion_decode
        from truesight.diffusion.schedule import Schedule
        from truesight.diffusion.train_diffusion import load_unet
        unet, ucfg = load_unet(a.diffusion, a.device); sched = Schedule(ucfg["T"], ucfg["kind"], device=a.device)

    rows, decodes, seam_items, seam_boxes = [], {"original": [], "coarse": [], "coarse+roi": [], "diffusion": [], "diffusion+roi": []}, [], {}
    for path in files:
        pil, boxes, text, faces, _ = eval_limits.limit_and_detect(Image.open(path).convert("RGB"), text_det, face_det, a.max_pixels)
        W, H = pil.size
        row = dict(image=Path(path).name, H=H, W=W, n_text=len(text), n_faces=len(faces), n_boxes=len(boxes),
                   area_fraction=roi.box_area_fraction(boxes, H, W))
        if a.draw:
            draw_boxes(pil, text, faces, boxes).save(figs / "boxes" / f"{Path(path).stem}.png")
        if use_codec:
            x = to_t(pil).to(a.device)
            blob, _ = roi.encode_protected(lo, hi, x, boxes)
            br = roi.size_breakdown(blob)
            row.update({k: br[k] for k in ("total_bytes", "roi_bytes", "roi_share", "bpp_total", "bpp_base", "bpp_roi")})
            p = roi.decode_protected(lo, hi, blob)
            coarse = p.coarse.clamp(0, 1)
            variants = {"coarse": coarse, "coarse+roi": roi.paste(coarse, p.canvas, p.soft_mask)}
            if unet is not None:
                dif = diffusion_decode(unet, sched, coarse, steps=a.steps, w=a.w)
                variants.update({"diffusion": dif, "diffusion+roi": roi.paste(dif, p.canvas, p.soft_mask)})
            decodes["original"].append(pil)
            for k, v in variants.items():
                decodes[k].append(to_pil(v))
            seam_items.append((Path(path).stem, to_pil(coarse), to_pil(variants["coarse+roi"]), boxes)); seam_boxes[Path(path).stem] = boxes
        rows.append(row)
        print(f"{row['image']}: {len(text)} text, {len(faces)} faces -> {len(boxes)} boxes, {row['area_fraction']:.0%} of the image"
              + (f", ROI {row['roi_share']:.0%} of {row['total_bytes']} bytes" if use_codec else ""), flush=True)

    df = pd.DataFrame(rows); df.to_csv(out / "boxes.csv", index=False)
    summary = {"set": a.set, "images": len(df), "mean_boxes": float(df.n_boxes.mean()),
               "images_with_boxes": int((df.n_boxes > 0).sum()), "mean_area_fraction": float(df.area_fraction.mean())}
    if use_codec:
        summary.update(mean_roi_share=float(df.roi_share.mean()), mean_bpp_total=float(df.bpp_total.mean()),
                       mean_bpp_base=float(df.bpp_base.mean()), mean_bpp_roi=float(df.bpp_roi.mean()))
        if a.draw:
            seam_figure(seam_items, seam_boxes, figs / "roi_seams.png")
    if a.tfr or a.fir:
        if not use_codec:
            sys.exit("--tfr/--fir compare decoded images: pass --codec-lo and --codec-hi")
        from truesight.protect.faithfulness import (EasyOCRAdapter, FaceEmbedder, face_identity_rate,
                                                    text_faithfulness_rate)
        methods = [m for m in ("coarse", "coarse+roi", "diffusion", "diffusion+roi") if decodes[m]]
        if a.tfr:
            ocr = EasyOCRAdapter()
            summary["tfr"] = {m: {k: v for k, v in text_faithfulness_rate(ocr, zip(decodes["original"], decodes[m])).items()
                                  if k != "per_image"} for m in methods}
        if a.fir:
            emb = FaceEmbedder()
            summary["fir"] = {m: {k: v for k, v in face_identity_rate(face_det, emb, zip(decodes["original"], decodes[m])).items()
                                  if k != "per_image"} for m in methods}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
