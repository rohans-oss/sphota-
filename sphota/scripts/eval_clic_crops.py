"""LPIPS and FID on matched 256 x 256 CLIC crops (section 8.1 / 3.4 of the guide: "FID on CLIC 2020 test crops, about 2,000 crops").

    python scripts/eval_clic_crops.py --fid                          # ~2,000 crops of the CLIC lists, lowest-lambda codec
    python scripts/eval_clic_crops.py --max-crops 16 --crops-per-image 2 --steps 4 --batch 4 --device cpu   # quick check

Every crop is coded with the lowest-lambda codec (estimated bits: no torchac needed), decoded by the codec alone (coarse), by the
diffusion decoder from the same bits, and compared with JPEG and WebP at the same bits per pixel (their real bpp is recorded;
``reachable_*`` says whether they could get that low). LPIPS is measured per crop; FID with clean-fid on the folders of original /
coarse / diffusion / jpeg / webp crops. The CLIC test labels are not public, so the CLIC *valid* lists stand in for the test split.

Resumable: after every batch the metrics are appended to <out>/rows.csv and the five crop images of the batch are stored in
<out>/batches/<first crop>.npz (small files that the run-state sync keeps), so a stop at the session time limit (exit status 75)
or a crash continues with the next unfinished crop, even on a new machine. The PNG folders for clean-fid are built from the
batches only when the FID is computed. Outputs in --out: rows.csv, batches/*.npz, summary.json, fid/ (deleted unless --keep-images).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from truesight import runtime  # noqa: E402
from truesight.baselines.jpeg import find_quality_for_bpp  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.decode import coarse_decode, diffusion_decode  # noqa: E402
from truesight.diffusion.eval_crops import (CROP, METHODS, build_summary, compute_fids, per_sample_bpp,  # noqa: E402
                                            plan_crops)
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402
from truesight.metrics import lpips_dist  # noqa: E402

PROC = ROOT / "data" / "processed"


def to_t(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


def save_batch(path, ids, stack):
    """One .npz per batch: crop ids and the uint8 images of every method, so a new session can rebuild the FID folders."""
    arrays = {m: np.stack([np.asarray(s[m]) for s in stack]) for m in ("original",) + METHODS}
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, ids=np.asarray(ids), **arrays)
    tmp.replace(path)


def resume_state(rows_path, bdir):
    """Crop ids that are finished: present in rows.csv AND stored in a batch file. Rows without images (a lost batch file) are
    dropped from rows.csv so the crops are recomputed instead of leaving a hole in the FID set."""
    if not rows_path.exists() or not rows_path.stat().st_size:
        return set()
    have = set()
    for f in sorted(bdir.glob("*.npz")):
        have |= {int(i) for i in np.load(f)["ids"]}
    rows = pd.read_csv(rows_path)
    keep = rows[rows["crop"].isin(have)]
    if len(keep) != len(rows):
        keep.to_csv(rows_path, index=False)
    return set(keep["crop"])


def write_png_folders(bdir, fid_root):
    """Unpack all batch files into fid/<method>/<crop>.png for clean-fid."""
    for m in ("original",) + METHODS:
        (fid_root / m).mkdir(parents=True, exist_ok=True)
    for f in sorted(bdir.glob("*.npz")):
        z = np.load(f)
        for m in ("original",) + METHODS:
            for k, im in zip(z["ids"], z[m]):
                Image.fromarray(im).save(fid_root / m / f"{int(k):05d}.png")


def default_w(path):
    """The CFG scale chosen by the Kodak sweep of Step 10 (summary.json), else the guide's 1.5."""
    p = Path(path)
    if p.exists():
        try:
            return float(json.loads(p.read_text())["default_w"])
        except (KeyError, ValueError):
            pass
    return 1.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", default=str(ROOT / "experiments" / "step7_codec" / "lmbda0.0018" / "final.pt"))
    ap.add_argument("--diffusion", default=str(ROOT / "experiments" / "step10_diffusion" / "final.pt"))
    ap.add_argument("--lists", nargs="*", default=[str(PROC / "test_clic.txt")])
    ap.add_argument("--crops-per-image", type=int, default=20)
    ap.add_argument("--max-crops", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--w", type=float, default=None, help="CFG scale (default: the one chosen by eval_diffusion.py, else 1.5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fid", action="store_true", help="also FID with clean-fid (downloads the Inception weights on first use)")
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step10_diffusion" / "clic_crops"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fid_root, bdir = out / "fid", out / "batches"
    bdir.mkdir(exist_ok=True)

    codec, ccfg = load_codec(a.codec, dev)
    unet, ucfg = load_unet(a.diffusion, dev)
    sched = Schedule(ucfg["T"], ucfg["kind"], device=dev)
    w = a.w if a.w is not None else default_w(Path(a.diffusion).parent / "eval" / "summary.json")
    files = [f for lst in a.lists if Path(lst).exists() for f in read_list(lst)]
    if not files:
        sys.exit("no CLIC images: run scripts/prepare_data.py first (or pass --lists)")
    sizes = []
    for f in files:
        with Image.open(f) as im:
            sizes.append((im.height, im.width))
    plan = plan_crops(sizes, a.crops_per_image, a.max_crops, CROP, a.seed)
    rows_path = out / "rows.csv"
    done = resume_state(rows_path, bdir)
    todo = [(k, *p) for k, p in enumerate(plan) if k not in done]
    print(f"{len(files)} images -> {len(plan)} crops of {CROP}x{CROP}; {len(done)} already done; w={w}, steps={a.steps}", flush=True)

    cache = {}
    for b0 in range(0, len(todo), a.batch):
        batch = todo[b0: b0 + a.batch]
        crops = []
        for k, i, y, x in batch:
            if i not in cache:
                cache.clear(); cache[i] = Image.open(files[i]).convert("RGB")
            crops.append(cache[i].crop((x, y, x + CROP, y + CROP)))
        xb = torch.cat([to_t(c) for c in crops]).to(dev)
        with torch.no_grad():
            bpp = per_sample_bpp(codec(xb)["likelihoods"], CROP * CROP).cpu().numpy()
            coarse = coarse_decode(codec, xb)
            dec = diffusion_decode(unet, sched, coarse, steps=a.steps, w=w, seed=a.seed + b0).cpu()
        coarse = coarse.cpu()
        rows, stack = [], []
        for j, ((k, i, y, x), pil) in enumerate(zip(batch, crops)):
            xt = to_t(pil)
            rec = dict(crop=k, image=Path(files[i]).name, y=y, x=x, bpp=float(bpp[j]))
            outs = {"coarse": to_pil(coarse[j:j + 1]), "diffusion": to_pil(dec[j:j + 1])}
            for fmt in ("jpeg", "webp"):
                _, b, r, reachable = find_quality_for_bpp(pil, float(bpp[j]), fmt.upper())
                outs[fmt] = r; rec[f"bpp_{fmt}"] = float(b); rec[f"reachable_{fmt}"] = int(bool(reachable))
            for m, im in outs.items():
                rec[f"lpips_{m}"] = lpips_dist(xt, to_t(im))
            stack.append({"original": pil, **outs})
            rows.append(rec)
        save_batch(bdir / f"{batch[0][0]:05d}.npz", [r["crop"] for r in rows], stack)           # images first, then the rows:
        pd.DataFrame(rows).to_csv(rows_path, mode="a", header=not rows_path.exists() or rows_path.stat().st_size == 0, index=False)
        print(f"crops {b0 + len(batch) + len(done)}/{len(plan)}: LPIPS coarse {np.mean([r['lpips_coarse'] for r in rows]):.3f} "
              f"diffusion {np.mean([r['lpips_diffusion'] for r in rows]):.3f}", flush=True)
        if b0 + a.batch < len(todo) and runtime.guard().stop_requested():
            raise runtime.Paused(f"[clic_crops] {runtime.guard().reason()}: {len(plan) - len(done) - b0 - len(batch)} crops left, run again")

    allrows = pd.read_csv(rows_path)
    fid = None
    if a.fid:
        try:
            write_png_folders(bdir, fid_root)
            fid = compute_fids(fid_root)
        except ImportError:
            sys.exit("--fid needs clean-fid: pip install clean-fid (it downloads Inception weights on first use)")
        print("FID:", fid, flush=True)
    summary = build_summary(allrows, fid, dict(codec_lmbda=ccfg["lmbda"], w=w, steps=a.steps, crop=CROP, images=len(files)))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("crops", "checks", "table") if k in summary}, indent=2, default=float))
    if not a.keep_images:
        shutil.rmtree(fid_root, ignore_errors=True)


if __name__ == "__main__":
    runtime.run_cli(main)
