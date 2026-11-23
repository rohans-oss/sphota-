"""Measure the diffusion decoder on this GPU and project the time of every evaluation stage (plan before you pay).

    python scripts/benchmark_decode.py                       # real network size, random weights (no checkpoint needed)
    python scripts/benchmark_decode.py --diffusion experiments/step10_diffusion/final.pt
    python scripts/benchmark_decode.py --max-pixels 0 --width 2048 --height 1408   # what a whole 2K CLIC image would cost

Times one U-Net call at 256 x 256 (CLIC crops) and at the evaluation window (default 768 x 512 = --max-pixels), then prints hours
per stage for the full-scale settings of scripts/run_full_pipeline.sh. Training is not included (that is iterations x the speed
printed in its log). Use it to decide --limit / --max-pixels / step counts before the long run.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from truesight.diffusion.benchmark import project, stage_plan, time_unet_call  # noqa: E402
from truesight.diffusion.unet import CHANNELS, make_unet, n_params  # noqa: E402
from truesight.eval_limits import MAX_PIXELS, window_size  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diffusion", default=None, help="final.pt (default: random weights of the guide's network)")
    ap.add_argument("--channels", type=int, nargs=4, default=list(CHANNELS))
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--width", type=int, default=768, help="width of a typical full-size image (for the window shape)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.diffusion:
        from truesight.diffusion.train_diffusion import load_unet
        unet, _ = load_unet(a.diffusion, dev)
    else:
        unet = make_unet(128, tuple(a.channels)).to(dev).eval()
    print(f"U-Net: {n_params(unet) / 1e6:.1f} M parameters on {dev}", flush=True)
    cw, ch = window_size(a.width, a.height, a.max_pixels)
    ch, cw = (ch + 63) // 64 * 64, (cw + 63) // 64 * 64                    # images are padded to a multiple of 64
    secs = {"crop": time_unet_call(unet, 256, 256, dev, a.repeats), "img": time_unet_call(unet, ch, cw, dev, a.repeats)}
    print(f"one U-Net call: {secs['crop'] * 1000:.0f} ms at 256x256, {secs['img'] * 1000:.0f} ms at {cw}x{ch}")
    print(f"one 50-step decode: {secs['crop'] * 100:.1f} s per 256x256 crop, {secs['img'] * 100:.1f} s per {cw}x{ch} image\n")
    rows, total = project(secs, stage_plan())
    print(f"{'stage':<18}{'decodes':>9}{'steps':>7}{'hours':>9}")
    for stage, n, steps, h in rows:
        print(f"{stage:<18}{n:>9}{steps:>7}{h:>9.1f}")
    print(f"{'TOTAL':<18}{'':>9}{'':>7}{total:>9.1f}  (U-Net calls only; the guided decodes also differentiate g_a)")


if __name__ == "__main__":
    main()
