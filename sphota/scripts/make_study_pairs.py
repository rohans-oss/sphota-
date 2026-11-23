"""Build the image pairs of the user study (section 8.5): Sphota against a baseline at the same file size.

    python scripts/make_study_pairs.py --codec-lo experiments/step7_codec/lmbda0.0018/final.pt \
        --codec-hi experiments/step7_codec/lmbda0.0483/final.pt --diffusion experiments/step10_diffusion/final.pt \
        --frozen experiments/step12_guidance/frozen.json --out study

For each image of the lists (default: Kodak, CLIC, text and face images, round-robin, up to --images) it encodes the protected
.sphota file, decodes it with the full Sphota decoder and with plain diffusion (same bits, no guidance, no ROI), and compresses
the original with JPEG and WebP to the same number of bits per pixel (or the lowest the format can reach: the real bpp of each
is recorded in pairs.csv). Pairs: Sphota vs jpeg, vs webp and vs plain. Writes into --out: images/<hash>.png (neutral file
names, so a participant cannot read the method from the URL) and pairs.csv (see truesight/analysis/user_study.py).
Needs torchac and the trained models.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from truesight import eval_limits, runtime  # noqa: E402
from truesight.baselines.jpeg import find_quality_for_bpp  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.guidance import sphota_decode  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402
from truesight.protect import roi  # noqa: E402
from truesight.protect.detect import FaceDetector, TextDetector  # noqa: E402

PROC = ROOT / "data" / "processed"
SETS = ["test_kodak", "test_clic", "test_text", "test_face"]


def to_t(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


def neutral_name(pair_id, role, salt):
    return hashlib.sha256(f"{salt}|{pair_id}|{role}".encode()).hexdigest()[:14] + ".png"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec-lo", required=True)
    ap.add_argument("--codec-hi", required=True)
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--frozen", default=str(ROOT / "experiments" / "step12_guidance" / "frozen.json"))
    ap.add_argument("--lists", nargs="*", default=[str(PROC / f"{s}.txt") for s in SETS])
    ap.add_argument("--images", type=int, default=40, help="images in the pool (taken round-robin from the lists)")
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    eval_limits.add_argument(ap)
    ap.add_argument("--salt", default="study")
    ap.add_argument("--out", default=str(ROOT / "study"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    fz = json.loads(Path(a.frozen).read_text())
    gk = dict(lr=fz["lr"], t_start=fz["t_start"], alpha=fz["alpha"], beta=fz["beta"], inner_steps=fz["inner_steps"], margin=fz["margin"])
    w, steps = a.w if a.w is not None else fz["w"], a.steps if a.steps is not None else fz["steps"]
    lo, _ = load_codec(a.codec_lo, dev); hi, _ = load_codec(a.codec_hi, dev)
    lo.prepare_for_coding(); hi.prepare_for_coding()
    unet, ucfg = load_unet(a.diffusion, dev); sched = Schedule(ucfg["T"], ucfg["kind"], device=dev)
    td, fd = TextDetector(), FaceDetector()

    pools = [read_list(p) for p in a.lists if Path(p).exists()]
    files = []
    for i in range(max((len(p) for p in pools), default=0)):                # round-robin over the sets
        for p in pools:
            if i < len(p) and len(files) < a.images:
                files.append(p[i])
    rows, pairs_csv = [], out / "pairs.csv"
    for n, path in enumerate(files):
        pil, boxes, _, _, _ = eval_limits.limit_and_detect(Image.open(path).convert("RGB"), td, fd, a.max_pixels)
        W, H = pil.size
        x = to_t(pil).to(dev)
        blob, _ = roi.encode_protected(lo, hi, x, boxes)
        prot = roi.decode_protected(lo, hi, blob)
        bpp = 8 * len(blob) / (W * H)
        full = to_pil(sphota_decode(lo, hi, unet, sched, prot=prot, w=w, steps=steps, guided=True, use_roi=True, **gk).x)
        plain = to_pil(sphota_decode(lo, hi, unet, sched, prot=prot, w=w, steps=steps, guided=False, use_roi=False).x)
        others = {"plain": (plain, bpp)}
        for fmt in ("jpeg", "webp"):
            _, b, rec, _ = find_quality_for_bpp(pil, bpp, fmt.upper())
            others[fmt] = (rec, b)
        for base, (img, obpp) in others.items():
            pid = f"{n:03d}_{base}"
            names = {r: neutral_name(pid, r, a.salt) for r in ("original", "sphota", "other")}
            for role, im in (("original", pil), ("sphota", full), ("other", img)):
                im.save(out / "images" / names[role])
            rows.append(dict(pair_id=pid, image=Path(path).name, baseline=base, original=f"images/{names['original']}",
                             sphota=f"images/{names['sphota']}", other=f"images/{names['other']}", bpp_sphota=bpp, bpp_other=obpp))
        pd.DataFrame(rows).to_csv(pairs_csv, index=False)
        print(f"{n + 1}/{len(files)} {Path(path).name}: {bpp:.3f} bpp, {len(boxes)} boxes", flush=True)
    print(f"wrote {len(rows)} pairs to {pairs_csv}")


if __name__ == "__main__":
    runtime.run_cli(main)
