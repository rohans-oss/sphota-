"""Step 12 tuning: choose the guidance strength (lr) and start step (t_start) on validation images, then freeze them.

    python scripts/tune_guidance.py --codec-lo experiments/step7_codec/lmbda0.0018/final.pt \
        --codec-hi experiments/step7_codec/lmbda0.0483/final.pt --diffusion experiments/step10_diffusion/final.pt

Grid: lr in {0.02, 0.05, 0.1} x t_start in {500, 700, 900} on 20 validation images (alpha = 1, beta = 10, 3 inner steps,
margin 0.05 stay fixed). For every configuration and for the unguided decode it records the consistency rate, LPIPS and
PSNR (and TFR / FIR with --tfr / --fir). The rule (see eval_guidance.choose_guidance_config): the highest consistency
among configurations whose LPIPS is at most 10% above the unguided decode. The result is written to
<out>/frozen.json; evaluate once on the test sets with scripts/eval_sphota.py --frozen <out>/frozen.json.
If consistency barely moves for every lr, the codec's Jacobian is small: extend the grid with --lr 0.2 0.5 1 2 5.
Needs torchac (real .sphota files): on Windows run from an x64 Native Tools prompt.
"""
import argparse
import itertools
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
from truesight.diffusion.eval_guidance import choose_guidance_config  # noqa: E402
from truesight.diffusion.guidance import consistency_rate, sphota_decode  # noqa: E402
from truesight.diffusion.sample import PrefixCache  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402
from truesight.metrics import lpips_dist, psnr  # noqa: E402
from truesight.protect import roi  # noqa: E402
from truesight import eval_limits  # noqa: E402
from truesight.protect.detect import FaceDetector, TextDetector  # noqa: E402

PROC = ROOT / "data" / "processed"


def to_t(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec-lo", required=True)
    ap.add_argument("--codec-hi", required=True)
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--list", default=str(PROC / "val.txt"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--lr", type=float, nargs="*", default=[0.02, 0.05, 0.1])
    ap.add_argument("--t-start", type=int, nargs="*", default=[500, 700, 900])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--inner-steps", type=int, default=3)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--tfr", action="store_true")
    ap.add_argument("--fir", action="store_true")
    eval_limits.add_argument(ap)
    ap.add_argument("--no-roi", action="store_true", help="tune without protected boxes")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step12_guidance"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

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

    items = []
    for path in read_list(a.list)[: a.limit]:
        pil, boxes, _, _, _ = eval_limits.limit_and_detect(Image.open(path).convert("RGB"), td, fd, a.max_pixels)
        boxes = [] if a.no_roi else boxes
        blob, _ = roi.encode_protected(lo, hi, to_t(pil).to(dev), boxes)
        items.append((pil, roi.decode_protected(lo, hi, blob)))
    print(f"{len(items)} validation images, {sum(1 for _, p in items if p.boxes)} with protected boxes", flush=True)

    prefixes = [PrefixCache() for _ in items]      # one per image: the grid settings share the steps before their guidance starts

    def evaluate(**kw):
        rows, pairs = [], []
        for (pil, prot), prefix in zip(items, prefixes):
            o = sphota_decode(lo, hi, unet, sched, prot=prot, w=a.w, steps=a.steps, prefix=prefix, **kw)
            x = to_t(pil).to(dev)
            rows.append(dict(consistency=consistency_rate(lo, o.x, prot.y_hat), lpips=lpips_dist(x.cpu(), o.x.cpu()),
                             psnr=psnr(x.cpu(), o.x.cpu())))
            pairs.append((pil, to_pil(o.x)))
        r = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        if ocr is not None:
            from truesight.protect.faithfulness import text_faithfulness_rate
            r["tfr"] = text_faithfulness_rate(ocr, pairs)["tfr"]
        if emb is not None:
            from truesight.protect.faithfulness import face_identity_rate
            r["fir"] = face_identity_rate(fd, emb, pairs)["fir"]
        return r

    base = evaluate(guided=False)
    print("unguided:", base, flush=True)
    grid = []
    for lr, t_start in itertools.product(a.lr, a.t_start):
        r = evaluate(guided=True, lr=lr, t_start=t_start, alpha=a.alpha, beta=a.beta, inner_steps=a.inner_steps)
        grid.append(dict(lr=lr, t_start=t_start, **r))
        print(f"lr {lr} t_start {t_start}: {r}", flush=True)
    df = pd.DataFrame(grid)
    df.to_csv(out / "tuning_grid.csv", index=False)
    cfg, why = choose_guidance_config(df, base["lpips"])
    frozen = dict(lr=cfg["lr"], t_start=cfg["t_start"], alpha=a.alpha, beta=a.beta, inner_steps=a.inner_steps, margin=0.05,
                  w=a.w, steps=a.steps, unguided=base, chosen=cfg, reason=why, n_images=len(items))
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2))
    print(f"\nFROZEN: lr={cfg['lr']} t_start={cfg['t_start']} - {why}")


if __name__ == "__main__":
    main()
