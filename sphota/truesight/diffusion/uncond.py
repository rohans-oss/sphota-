"""Step 9 warm-up: an unconditional 64 x 64 DDPM trained with our own schedule, forward process and DDIM sampler.

    python scripts/step9_ddpm.py                                  # about 20k iterations (guide), then both figures
    python scripts/step9_ddpm.py --iters 300 --batch 8 --channels 16 32 32 32 --workers 0 --samples 4   # CPU smoke run

Why it exists: it shows that the forward process (``Schedule.q_sample``), the eps-prediction loss and the DDIM sampler
(``ddim_sample_eps``) are our own and work, before they are used inside the conditional decoder of Step 10. The network is
the same diffusers U-Net family with 3 input channels (noisy image only, no condition) and the same training recipe as the
main run: AdamW, warm-up + cosine learning rate, EMA, fp16 autocast on a GPU, gradient clipping, checkpoints every 5k
steps, resume from last.pt, crash-safe and time-budget aware (see truesight/runtime.py).

Outputs: <out>/log.csv, last.pt, final.pt, summary.json; <figs>/step9_forward_process.png (x_0 -> noise at t = 0, 250, 500,
750, 999) and <figs>/step9_samples.png (a grid of DDIM samples). summary.json also holds the check from the guide:
samples must be "recognisable blobs or textures, not pure noise or flat grey" (``sample_checks``).
"""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import runtime, speed
from ..data import CropDataset
from .sample import ddim_sample_eps
from .schedule import Schedule
from .train_diffusion import _atomic_save, drop_condition, lr_at, seed_all, update_ema  # noqa: F401
from .unet import make_unet, unet_config

ROOT = Path(__file__).resolve().parents[2]
FORWARD_TS = (0, 250, 500, 750, 999)
CHANNELS = (64, 128, 128, 256)          # smaller than the conditional decoder: 64 x 64 crops, 3 input channels


def load_uncond(path, device="cpu", use_ema=True):
    s = torch.load(path, map_location=device)
    cfg = s["config"]
    net = make_unet(in_channels=3, **cfg["unet"]).to(device)
    net.load_state_dict(s["ema"] if use_ema else s["unet"])
    return net.eval(), cfg


def train_uncond(out, iters=20_000, batch=64, crop=64, lr=2e-4, weight_decay=1e-4, warmup=500, ema_decay=0.999,
                 channels=CHANNELS, T=1000, kind="cosine", clip=1.0, ckpt_every=5000, log_every=100, workers=4, seed=0,
                 train_list=None, device=None, amp=None, stop_at=None):
    """Train (or resume) the unconditional DDPM. Raises ``runtime.Paused`` after saving when the time budget is used up and
    saves last.pt before re-raising any crash."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda") if amp is None else (amp and device.type == "cuda")
    speed.setup(device)                                           # cudnn.benchmark: only changes speed
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    sched = Schedule(T, kind, device=device)
    cfg = {"unet": unet_config(crop, channels), "T": T, "kind": kind, "ema_decay": ema_decay}
    unet = make_unet(in_channels=3, **cfg["unet"]).to(device)
    ema = copy.deepcopy(unet).eval().requires_grad_(False)
    opt = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    step, last = 0, out / "last.pt"
    if last.exists():
        s = torch.load(last, map_location=device)
        unet.load_state_dict(s["unet"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"]); step = s["step"]
        seed_all(seed + step)
        print(f"[uncond] resumed from step {step}", flush=True)
    loader = torch.utils.data.DataLoader(CropDataset(train_list or ROOT / "data" / "processed" / "train.txt", crop),
                                         batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                                         persistent_workers=workers > 0)

    def stream():
        while True:
            yield from loader

    batches = stream()
    new_log = not (out / "log.csv").exists()
    logf = open(out / "log.csv", "a", newline="")
    w = csv.writer(logf)
    if new_log:
        w.writerow(["step", "loss", "lr"]); logf.flush()
    unet.train()
    t0, stop = time.time(), (iters if stop_at is None else min(iters, stop_at))
    guard = runtime.guard()

    def save():
        _atomic_save({"unet": unet.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                      "scaler": scaler.state_dict(), "step": step, "config": cfg}, last)
        guard.saved()

    with runtime.crash_safe(save, "uncond", lambda: runtime.finite_params(unet, ema)):
        while step < stop:
            cur_lr = lr_at(step, iters, warmup, lr)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            x0 = next(batches).to(device, non_blocking=True) * 2 - 1
            t = torch.randint(0, T, (x0.size(0),), device=device)
            eps = torch.randn_like(x0)
            xt = sched.q_sample(x0, t, eps)                         # forward process (our own code)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                eps_hat = unet(xt, t).sample
            loss = F.mse_loss(eps_hat.float(), eps)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            update_ema(ema, unet, ema_decay)
            step += 1
            if step % log_every == 0:
                w.writerow([step, loss.item(), cur_lr]); logf.flush()
                print(f"[uncond] step {step}/{iters} loss {loss.item():.4f} lr {cur_lr:.2e} {time.time() - t0:.0f}s", flush=True)
            if step % ckpt_every == 0 or step == stop or guard.save_due():
                save()
            if step < stop:
                runtime.pause_if_needed(save, "uncond")
    logf.close()
    if step >= iters:
        _atomic_save({"unet": unet.state_dict(), "ema": ema.state_dict(), "step": step, "config": cfg}, out / "final.pt")
    return ema


@torch.no_grad()
def sample_uncond(unet, sched, n=16, size=64, steps=50, seed=0, device="cpu"):
    """n unconditional DDIM samples (n, 3, size, size) in [-1, 1]."""
    return ddim_sample_eps(lambda x, tt: unet(x, tt).sample, sched, (n, 3, size, size), steps=steps, seed=seed, device=device)


def sample_checks(samples, min_std=0.05, min_neighbour_corr=0.5):
    """The guide's check: samples are structure, not pure noise (neighbouring pixels correlate) and not flat grey (the image
    has contrast). ``samples`` is (n, 3, H, W) in [-1, 1]. Returns per-set statistics and the two verdicts."""
    s = samples.float().cpu()
    std = s.flatten(1).std(dim=1)
    a, b = s[..., :, :-1].flatten(1), s[..., :, 1:].flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
    return dict(mean_std=float(std.mean()), mean_neighbour_corr=float(corr.mean()),
                not_flat=bool((std > min_std).float().mean() >= 0.8),
                not_pure_noise=bool(corr.mean() >= min_neighbour_corr))


def forward_process_figure(x0, sched, path, ts=FORWARD_TS, seed=0):
    """One image noised at t = 0, 250, 500, 750, 999 with the same eps (so the columns show the same image dissolving)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0 = x0.to(sched.device)
    eps = torch.randn(x0.shape, generator=g).to(sched.device)
    fig, ax = plt.subplots(1, len(ts), figsize=(2.6 * len(ts), 3.0))
    for a, t in zip(ax, ts):
        xt = sched.q_sample(x0, torch.tensor([t], device=sched.device), eps)
        a.imshow(((xt[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()); a.axis("off")
        a.set_title(f"t = {t}   (alpha_bar {float(sched.ab[t]):.3f})", fontsize=8)
    fig.suptitle(f"Forward process ({sched.kind} schedule): x_t = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) eps")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def sample_grid_figure(samples, path, cols=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = samples.size(0)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows), squeeze=False)
    for i, a in enumerate(ax.ravel()):
        a.axis("off")
        if i < n:
            a.imshow(((samples[i].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy())
    fig.suptitle("Unconditional DDPM samples (64 x 64, DDIM)")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step9_ddpm"))
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--channels", type=int, nargs=4, default=list(CHANNELS))
    ap.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps for the sample grid")
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    train_uncond(out, iters=a.iters, batch=a.batch, crop=a.crop, lr=a.lr, channels=tuple(a.channels), kind=a.schedule,
                 ckpt_every=a.ckpt_every, workers=a.workers, seed=a.seed, train_list=a.train_list, device=a.device)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    unet, cfg = load_uncond(out / "final.pt", dev)
    sched = Schedule(cfg["T"], cfg["kind"], device=dev)
    x = CropDataset(a.train_list or ROOT / "data" / "processed" / "train.txt", a.crop)[0][None] * 2 - 1
    forward_process_figure(x, sched, Path(a.figs) / "step9_forward_process.png")
    samples = sample_uncond(unet, sched, n=a.samples, size=a.crop, steps=a.steps, seed=a.seed, device=dev)
    sample_grid_figure(samples, Path(a.figs) / "step9_samples.png")
    chk = sample_checks(samples)
    summary = dict(iters=a.iters, schedule=cfg["kind"], checks=chk,
                   abar=dict(first=float(sched.ab[0]), last=float(sched.ab[-1])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    runtime.run_cli(main)
