"""Check that the evaluation speed-ups give the same decodes before the evaluation stages use them (about 5-10 minutes on a T4).

    python scripts/eval_parity.py --codec experiments/step7_codec/lmbda0.0018/final.pt --diffusion experiments/step10_diffusion/final.pt

The evaluation stages decode thousands of images with DDIM + classifier-free guidance. Two changes make that faster without changing
what is computed beyond float noise:
    batch_cfg   the conditional and the unconditional U-Net passes run as ONE batch of 2 instead of two calls
    amp         the U-Net runs under fp16 autocast (it was trained that way; evaluation used fp32 before)
Both are used only after this check passed (TS_EVAL_AMP / TS_EVAL_BATCH_CFG = auto, the default, read the result file). On a few
test images, centre crops, the same seed and settings, it decodes with the fp32 reference and with each candidate and compares:

    batch_ok   PSNR(reference, batched fp32 decode) >= 60 dB        (the maths is identical: only float noise remains)
    amp_ok     PSNR(reference, fp16 decode) >= 35 dB, AND at least 10 dB above the PSNR between two SEEDS of the fp32 decoder (the
               sampling noise the metrics already live with), AND the paired LPIPS-to-original difference averages <= 0.01
Also checked (information only, they are exact by construction and unit-tested): resuming a decode from a shared prefix and sharing the
first step across CFG scales give the same image as decoding from scratch. Without a CUDA GPU nothing is enabled. The decision and the
numbers go to --out (default experiments/eval_parity.json); delete it to go back to the fp32 reference path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from truesight import runtime  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.data import read_list  # noqa: E402
from truesight.diffusion.decode import coarse_decode, diffusion_decode  # noqa: E402
from truesight.diffusion.sample import FirstStepCache, PrefixCache  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402
from truesight.diffusion.train_diffusion import load_unet  # noqa: E402
from truesight.metrics import lpips_dist  # noqa: E402

BATCH_MIN_PSNR = 60.0
AMP_MIN_PSNR = 35.0
AMP_MARGIN_DB = 10.0
AMP_MAX_LPIPS_DELTA = 0.01


def psnr_between(a, b):
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def decide(batch_psnr, amp_psnr, seed_psnr, lpips_delta, cuda=True):
    """The decision rule (pure function, unit tested). Inputs are per-image lists / the mean paired LPIPS difference."""
    batch_ok = bool(cuda and min(batch_psnr) >= BATCH_MIN_PSNR)
    amp_ok = bool(cuda and np.mean(amp_psnr) >= AMP_MIN_PSNR and np.mean(amp_psnr) >= np.mean(seed_psnr) + AMP_MARGIN_DB
                  and abs(lpips_delta) <= AMP_MAX_LPIPS_DELTA)
    return {"batch_ok": batch_ok, "amp_ok": amp_ok}


class _Guide:
    """A stand-in for the faithfulness guide: changes x0_hat for t <= t_start (like FaithfulGuide), so prefix reuse has to be exact."""
    k, alpha, roi_mask = 1, 1.0, None

    def __init__(self, t_start=700):
        self.t_start = t_start

    def __call__(self, x0, t):
        return x0 if t > self.t_start else x0 * 0.98


def center_crop(path, size):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(size, (min(w, h) // 64) * 64)
    x0, y0 = (w - s) // 2, (h - s) // 2
    a = np.asarray(im.crop((x0, y0, x0 + s, y0 + s)), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", default=str(ROOT / "experiments" / "step7_codec" / "lmbda0.0018" / "final.pt"))
    ap.add_argument("--diffusion", default=str(ROOT / "experiments" / "step10_diffusion" / "final.pt"))
    ap.add_argument("--test-list", default=str(ROOT / "data" / "processed" / "test_kodak.txt"))
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "eval_parity.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    codec, _ = load_codec(a.codec, dev)
    unet, ucfg = load_unet(a.diffusion, dev)
    sched = Schedule(ucfg["T"], ucfg["kind"], device=dev)
    files = read_list(a.test_list)[: a.images]
    batch_psnr, amp_psnr, seed_psnr, lp_ref, lp_amp, t_ref, t_fast = [], [], [], [], [], 0.0, 0.0
    prefix_err = first_err = 0.0

    def dec(coarse, **kw):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = diffusion_decode(unet, sched, coarse, steps=a.steps, w=a.w, **kw)
        if cuda:
            torch.cuda.synchronize()
        return y, time.perf_counter() - t0

    for f in files:
        x = center_crop(f, a.size).to(dev)
        coarse = coarse_decode(codec, x, amp=False)
        ref, dt = dec(coarse, seed=0, amp=False, batch_cfg=False); t_ref += dt
        bat, _ = dec(coarse, seed=0, amp=False, batch_cfg=True)
        fast, dt = dec(coarse, seed=0, amp=True, batch_cfg=True); t_fast += dt
        other, _ = dec(coarse, seed=1, amp=False, batch_cfg=False)
        batch_psnr.append(psnr_between(ref, bat)); amp_psnr.append(psnr_between(ref, fast)); seed_psnr.append(psnr_between(ref, other))
        lp_ref.append(lpips_dist(x.cpu(), ref.cpu())); lp_amp.append(lpips_dist(x.cpu(), fast.cpu()))
        # exactness of the reuse machinery (fp32, same settings): prefix resume and first-step sharing vs from scratch
        g1, g2 = _Guide(), _Guide()
        scratch, _ = dec(coarse, seed=0, amp=False, batch_cfg=False, guide=g1)
        pc = PrefixCache()
        dec(coarse, seed=0, amp=False, batch_cfg=False, prefix=pc)                       # records the shared (unguided) path
        resumed, _ = dec(coarse, seed=0, amp=False, batch_cfg=False, guide=g2, prefix=pc)
        prefix_err = max(prefix_err, float((scratch - resumed).abs().max()))
        fs = FirstStepCache()
        dec(coarse, seed=0, amp=False, batch_cfg=False, first_step=fs)
        shared, _ = dec(coarse, seed=0, amp=False, batch_cfg=False, first_step=fs)
        first_err = max(first_err, float((ref - shared).abs().max()))
        print(f"{Path(f).name}: batched {batch_psnr[-1]:.1f} dB, fp16 {amp_psnr[-1]:.1f} dB, seed-to-seed {seed_psnr[-1]:.1f} dB", flush=True)

    d = decide(batch_psnr, amp_psnr, seed_psnr, float(np.mean(lp_amp) - np.mean(lp_ref)), cuda)
    d.update(device=str(dev), images=len(files), steps=a.steps, w=a.w, size=a.size, batch_psnr_db=batch_psnr, amp_psnr_db=amp_psnr,
             seed_to_seed_psnr_db=seed_psnr, lpips_ref=float(np.mean(lp_ref)), lpips_amp=float(np.mean(lp_amp)),
             speedup_amp_batch=(t_ref / t_fast) if t_fast > 0 else None, prefix_resume_max_abs_err=prefix_err,
             first_step_share_max_abs_err=first_err, prefix_ok=bool(prefix_err < 1e-4), first_step_ok=bool(first_err < 1e-4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2))
    print(json.dumps({k: v for k, v in d.items() if not isinstance(v, list)}, indent=2))
    if not cuda:
        print("NOTE: no CUDA GPU: nothing is enabled (the fp16 / batched paths only exist on a GPU).")
    print(f"batched guidance passes: {'ENABLED' if d['batch_ok'] else 'off'};  fp16 U-Net decoding: {'ENABLED' if d['amp_ok'] else 'off'}")


if __name__ == "__main__":
    runtime.run_cli(main)
