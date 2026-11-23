"""End-to-end check of Step 1 (data) + Step 2 (metrics).

1. Loads the split lists written by prepare_data.py and verifies the Step-1 done-criteria.
2. Pulls real Kodak images through FullImageDataset, encodes JPEG and WebP at several qualities,
   measures bpp / PSNR / MS-SSIM / LPIPS and BD-rate (WebP vs JPEG should be clearly negative).
3. Checks CropDataset yields the right shapes from train.txt.

    python scripts/e2e_check.py [--quality 10 20 30 50 70 90]
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.data import CropDataset, FullImageDataset, read_list  # noqa: E402
from truesight.metrics import bd_rate, bpp_from_bytes, lpips_dist, msssim, psnr  # noqa: E402

PROC = ROOT / "data" / "processed"
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def codec(img: Image.Image, fmt: str, q: int):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=q)
    n = buf.tell()
    buf.seek(0)
    return Image.open(buf).convert("RGB"), n


def to_t(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", nargs="*", type=int, default=[10, 20, 30, 50, 70, 90])
    a = ap.parse_args()

    # ---- Step 1: lists and leakage
    lists = {p.stem: read_list(p) for p in sorted(PROC.glob("*.txt"))}
    print({k: len(v) for k, v in lists.items()})
    missing = [f for v in lists.values() for f in v if not Path(f).exists()]
    check("every listed file exists", not missing, f"({len(missing)} missing)")
    train = set(lists.get("train", []))
    for k, v in lists.items():
        if k != "train":
            check(f"train and {k} do not overlap", not (train & set(v)))
    check("train has >= 3000 images", len(train) >= 3000, f"({len(train)})")
    check("Kodak has 24 images", len(lists.get("test_kodak", [])) == 24)

    # ---- Step 1: datasets
    if train:
        x = CropDataset(PROC / "train.txt", 256)[0]
        check("CropDataset sample is (3,256,256) in [0,1]", x.shape == (3, 256, 256) and 0 <= x.min() <= x.max() <= 1)
    ds = FullImageDataset(PROC / "test_kodak.txt")
    img, (h, w), path = ds[0]
    check("FullImageDataset pads to multiple of 64", img.shape[1] % 64 == 0 and img.shape[2] % 64 == 0, f"{(h, w)} -> {tuple(img.shape[1:])}")

    # ---- Step 2 on real Kodak data
    rows = {"JPEG": [], "WebP": []}
    for fmt in rows:
        for q in a.quality:
            acc = {"bpp": [], "psnr": [], "msssim": [], "lpips": []}
            for i in range(len(ds)):
                ref = Image.open(read_list(PROC / "test_kodak.txt")[i]).convert("RGB")
                rec, n = codec(ref, fmt, q)
                x, y = to_t(ref), to_t(rec)
                acc["bpp"].append(bpp_from_bytes(n, ref.height, ref.width))
                acc["psnr"].append(psnr(x, y)); acc["msssim"].append(msssim(x, y)); acc["lpips"].append(lpips_dist(x, y))
            rows[fmt].append({k: float(np.mean(v)) for k, v in acc.items()})
            r = rows[fmt][-1]
            print(f"{fmt:5s} q={q:3d}  bpp={r['bpp']:.3f}  PSNR={r['psnr']:.2f}  MS-SSIM={r['msssim']:.4f}  LPIPS={r['lpips']:.4f}")
    for fmt, rs in rows.items():
        b = [r["bpp"] for r in rs]; p = [r["psnr"] for r in rs]; l = [r["lpips"] for r in rs]
        check(f"{fmt}: bpp increases with quality", all(np.diff(b) > 0))
        check(f"{fmt}: PSNR increases with quality", all(np.diff(p) > 0))
        check(f"{fmt}: LPIPS decreases with quality", all(np.diff(l) < 0))
    j, w_ = rows["JPEG"], rows["WebP"]
    bdr = bd_rate([r["bpp"] for r in j], [r["psnr"] for r in j], [r["bpp"] for r in w_], [r["psnr"] for r in w_])
    print(f"BD-rate WebP vs JPEG (PSNR, Kodak): {bdr:.1f}%")
    check("WebP saves bits vs JPEG (BD-rate < 0; literature ~ -20..-35%)", bdr < 0)
    check("BD-rate of JPEG vs itself is 0", abs(bd_rate([r['bpp'] for r in j], [r['psnr'] for r in j], [r['bpp'] for r in j], [r['psnr'] for r in j])) < 1e-6)
    # ---- Step 3 artefacts: RD curves exist on Kodak and the JPEG size floor is recorded
    import json
    s3 = ROOT / "experiments" / "step3_baselines" / "summary.json"
    check("Step 3 summary + RD plot exist", s3.exists() and (ROOT / "report/figures/rd_kodak.png").exists())
    if s3.exists():
        fl = json.loads(s3.read_text())["kodak"]["size_floor"]["JPEG"]["min_bpp_mean"]
        check("Step 3: JPEG size floor recorded and plausible (0.1-0.3 bpp)", 0.1 < fl < 0.3, f"({fl:.3f} bpp)")
    # ---- Step 4 artefacts: PCA + VQ on the same plot as JPEG, PCA variance figure
    s4 = ROOT / "experiments" / "step4_baselines" / "summary.json"
    check("Step 4 summary + combined RD plot + PCA variance figure exist",
          s4.exists() and (ROOT / "report/figures/rd_kodak_all.png").exists()
          and (ROOT / "report/figures/pca_explained_variance.png").exists())
    if s4.exists():
        import pandas as pd
        d = json.loads(s4.read_text())
        check("Step 4: >= 4 RD points for PCA and VQ", all(d["points"].get(m, 0) >= 4 for m in ("PCA-16", "PCA-64", "VQ")), str(d["points"]))
        check("Step 4: first 8 PCA components hold > 90% of the variance", d["pca_variance"]["top8_pct"] > 90,
              f"({d['pca_variance']['top8_pct']:.1f}%)")
        r4 = pd.read_csv(ROOT / "experiments" / "step4_baselines" / "rd_results.csv")
        check("Step 4: 24 Kodak images evaluated", r4.image.nunique() == 24)
        vq = r4[r4.method == "VQ"].groupby("quality").bpp.mean().sort_index()
        check("Step 4: VQ bpp grows with codebook size", all(np.diff(vq.values) > 0))
    # ---- Step 5: features CSV covers every listed image; R2 target; regularisation plot
    import pandas as pd
    fcsv = PROC / "features.csv"
    check("Step 5: features.csv exists", fcsv.exists())
    if fcsv.exists():
        ft = pd.read_csv(fcsv)
        listed = {p.relative_to(ROOT).as_posix() if Path(p).is_absolute() else p
                  for p in (l for f in PROC.glob("*.txt") for l in f.read_text().splitlines() if l.strip())}
        check("Step 5: features for ALL train/val/test images (incl. names with spaces)",
              ft.ok.astype(bool).all() and set(ft.path) == listed, f"({len(ft)} rows vs {len(listed)} listed)")
        check("Step 5: no NaN/inf in features", np.isfinite(ft[["grad_mean", "lap_var", "entropy", "edge_density", "colorfulness", "brightness", "jpeg50_bpp", "jpeg10_bpp"]].to_numpy()).all())
    s5 = ROOT / "experiments" / "step5_size_predictor" / "results.json"
    check("Step 5: results.json + regularisation plot exist", s5.exists() and (ROOT / "report/figures/size_predictor_regularisation.png").exists())
    if s5.exists():
        r5 = json.loads(s5.read_text())
        check("Step 5: best nested-CV R2 >= 0.6", r5["best_guide_model"]["cv_r2_mean"] >= 0.6, f"({r5['best_guide_model']})")
        check("Step 5: Lasso coefficients reported", len(r5["variants"]["guide (7 raw features)"]["Lasso"]["coefficients"]) == 7)
    print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
