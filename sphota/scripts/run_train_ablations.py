"""Ablations A7 (linear vs cosine noise schedule) and A8 (Adam vs AdamW, constant vs cosine LR).

    python scripts/run_train_ablations.py --codec experiments/step7_codec/lmbda0.0018/final.pt            # 4 runs x 20k iterations
    python scripts/run_train_ablations.py --codec ... --iters 200 --batch 4 --channels 16 32 32 32 --workers 0 --crop 64 \
                                          --eval-images 2 --eval-crop 128 --steps 4                       # CPU smoke run

Trains the reference run (cosine schedule, AdamW, cosine LR) and three variants that each change one thing, all with the same
number of iterations (--iters, default 20,000: much shorter than the 200k-step main run, so absolute numbers are worse than
the main model; only the comparison between the four matters), then decodes the same Kodak images with each model and writes
<out>/results.csv, summary.csv and summary.json. Every run resumes from its own last.pt and is skipped when final.pt exists.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight import runtime  # noqa: E402
import torch  # noqa: E402
import pandas as pd  # noqa: E402
from truesight.diffusion.ablation import ABLATIONS, evaluate_unet, summarize_ablations, train_loss_tail  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet, train_diffusion  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.unet import CHANNELS  # noqa: E402

PROC = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", default=str(ROOT / "experiments" / "step7_codec" / "lmbda0.0018" / "final.pt"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step10_ablation"))
    ap.add_argument("--runs", nargs="*", default=list(ABLATIONS), choices=list(ABLATIONS))
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--channels", type=int, nargs=4, default=list(CHANNELS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--test-list", default=str(PROC / "test_kodak.txt"))
    ap.add_argument("--eval-images", type=int, default=24)
    ap.add_argument("--eval-crop", type=int, default=None, help="centre-crop the evaluation images (CPU runs)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    codec, _ = load_codec(a.codec, dev)
    files = read_list(a.test_list)
    results, tails = {}, {}
    for name in a.runs:
        _, kw = ABLATIONS[name]
        d = out / name
        if not (d / "final.pt").exists():
            print(f"=== training {name} ({ABLATIONS[name][0]}): {kw or 'reference'}", flush=True)
            train_diffusion(a.codec, d, iters=a.iters, batch=a.batch, crop=a.crop, channels=tuple(a.channels),
                            workers=a.workers, ckpt_every=a.ckpt_every, train_list=a.train_list, **kw)
        unet, ucfg = load_unet(d / "final.pt", dev)
        sched = Schedule(ucfg["T"], ucfg["kind"], device=dev)
        df = evaluate_unet(unet, sched, codec, files, steps=a.steps, w=a.w, device=dev, max_images=a.eval_images,
                           crop=a.eval_crop)
        df.insert(0, "run", name); df.to_csv(d / "eval.csv", index=False)
        results[name], tails[name] = df, train_loss_tail(d / "log.csv")
        print(f"{name}: LPIPS {df.lpips.mean():.4f} PSNR {df.psnr.mean():.2f}", flush=True)
    pd.concat(results.values(), ignore_index=True).to_csv(out / "results.csv", index=False)
    summ = summarize_ablations(results)
    summ["train_loss_tail"] = summ["run"].map(tails)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ.to_dict("records"), indent=2, default=float))
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    runtime.run_cli(main)
