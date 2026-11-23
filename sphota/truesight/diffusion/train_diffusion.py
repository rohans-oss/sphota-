"""DDPM training of the conditional decoder (Step 10).

    python -m truesight.diffusion.train_diffusion --codec experiments/step7_codec/lmbda0.0018/final.pt \
           --out experiments/step10_diffusion

The lowest-lambda codec (0.0018) is frozen; its decode of the training crop (from the rounded latent) is the condition.
Images and conditions are scaled to [-1, 1]; the condition is replaced by zeros 10% of the time so that classifier-free
guidance works at sampling time; the network predicts eps and is trained with MSE. AdamW (1e-4) with 2k warm-up steps
then cosine decay, fp16 autocast + GradScaler on a GPU, gradient clipping, EMA weights (0.999), batch 32 of 128x128 crops,
about 200k steps. A checkpoint is written every 5k steps and training resumes from <out>/last.pt automatically.
Writes <out>/log.csv (step, loss, lr) and <out>/final.pt (EMA weights + raw weights + config).
"""
import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import runtime, speed
from ..codec.train_codec import load_codec
from ..data import CropDataset, to_float_batch
from .decode import coarse_decode
from .schedule import Schedule
from .unet import CHANNELS, make_unet, unet_config

ROOT = Path(__file__).resolve().parents[2]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def lr_at(step, total, warmup, base_lr):
    """Linear warm-up to ``base_lr`` over ``warmup`` steps, then cosine decay to 0 at ``total``."""
    warm = min(1.0, step / warmup) if warmup > 0 else 1.0
    return base_lr * warm * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def drop_condition(c, p, generator=None):
    """Classifier-free-guidance dropout: zero the condition of each sample with probability ``p``."""
    keep = (torch.rand(c.size(0), 1, 1, 1, device=c.device, generator=generator) >= p).to(c.dtype)
    return c * keep


@torch.no_grad()
def update_ema(ema, net, decay=0.999):
    """ema <- decay * ema + (1 - decay) * net (parameters only; the U-Net has no running statistics).
    One fused ``_foreach`` call over all parameters (same arithmetic, one multiply and one add each, as the plain loop)."""
    pe, pm = list(ema.parameters()), [p.detach() for p in net.parameters()]
    if hasattr(torch, "_foreach_mul_"):
        torch._foreach_mul_(pe, decay)
        torch._foreach_add_(pe, pm, alpha=1 - decay)
    else:
        for a, b in zip(pe, pm):
            a.mul_(decay).add_(b, alpha=1 - decay)


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)                          # a kill during the write never corrupts the previous checkpoint


def load_unet(path, device="cpu", use_ema=True):
    """Rebuild the network from a final.pt / last.pt checkpoint. Returns (unet in eval mode, config dict)."""
    s = torch.load(path, map_location=device)
    cfg = s["config"]
    net = make_unet(**cfg["unet"]).to(device)
    net.load_state_dict(s["ema"] if use_ema else s["unet"])
    return net.eval(), cfg


def train_diffusion(codec_path, out, iters=200_000, batch=32, crop=128, lr=1e-4, weight_decay=1e-4, warmup=2000,
                    ema_decay=0.999, p_drop=0.1, channels=CHANNELS, T=1000, kind="cosine", clip=1.0, ckpt_every=5000,
                    log_every=100, workers=4, seed=0, train_list=None, device=None, amp=None, stop_at=None,
                    optimizer="adamw", lr_schedule="cosine"):
    """Train (or resume) the conditional decoder. ``stop_at`` stops early after saving, used to test resume.
    ``kind`` (noise schedule), ``optimizer`` ("adamw" | "adam") and ``lr_schedule`` ("cosine" | "constant") are the three
    settings of ablations A7 / A8. Raises ``runtime.Paused`` after saving when the session time budget is used up, and writes
    last.pt before re-raising if anything crashes, so the run always continues from where it stopped."""
    if optimizer not in ("adamw", "adam") or lr_schedule not in ("cosine", "constant"):
        raise ValueError(f"optimizer must be adamw|adam and lr_schedule cosine|constant, got {optimizer!r}, {lr_schedule!r}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda") if amp is None else (amp and device.type == "cuda")
    speed.setup(device)                                           # cudnn.benchmark (fixed crop size): only changes speed
    use_cl = speed.use_channels_last("unet", device, use_amp)      # channels_last pairs with fp16 autocast on Tensor Cores
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    codec, codec_cfg = load_codec(codec_path, device)             # frozen: its decode of x is the condition
    codec.requires_grad_(False)
    sched = Schedule(T, kind, device=device)
    cfg = {"unet": unet_config(crop, channels), "T": T, "kind": kind, "codec_lmbda": codec_cfg["lmbda"],
           "p_drop": p_drop, "ema_decay": ema_decay, "optimizer": optimizer, "lr_schedule": lr_schedule}
    unet = make_unet(**cfg["unet"]).to(device)
    ema = copy.deepcopy(unet).eval().requires_grad_(False)
    if use_cl:
        speed.to_channels_last(unet, ema)                         # the checkpoint format (state_dict values) is unchanged
    fused = speed.fused_ok(device)                                # one fused kernel for the optimiser step: same update rule
    opt = (torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay, fused=fused) if optimizer == "adamw"
           else torch.optim.Adam(unet.parameters(), lr=lr, fused=fused))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    step, last = 0, out / "last.pt"
    if last.exists():                                              # resume
        s = torch.load(last, map_location=device)
        unet.load_state_dict(s["unet"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"]); step = s["step"]
        seed_all(seed + step)
        print(f"resumed from step {step}", flush=True)

    loader = torch.utils.data.DataLoader(CropDataset(train_list or ROOT / "data" / "processed" / "train.txt", crop, raw=True),
                                         batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                                         persistent_workers=workers > 0, **speed.loader_kwargs(workers, device))
    meter = speed.WaitMeter()                                     # how long the GPU waits for data (see the log line)

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

    def save():                                                    # reads the loop's current ``step`` and weights
        _atomic_save({"unet": unet.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                      "scaler": scaler.state_dict(), "step": step, "config": cfg}, last)
        guard.saved()

    with runtime.crash_safe(save, "diffusion", lambda: runtime.finite_params(unet, ema)):
        while step < stop:
            cur_lr = lr if lr_schedule == "constant" else lr_at(step, iters, warmup, lr)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            with meter:
                raw = next(batches)
            x = to_float_batch(raw, device)                        # uint8 -> float and the random flip, on the GPU
            x0 = x * 2 - 1                                         # images in [-1, 1]
            c = coarse_decode(codec, x) * 2 - 1                    # condition in [-1, 1] (rounded-latent decode)
            c = drop_condition(c, p_drop)                          # null condition (zeros) for CFG
            t = torch.randint(0, T, (x.size(0),), device=device)
            eps = torch.randn_like(x0)
            xt = sched.q_sample(x0, t, eps)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                eps_hat = unet(speed.cl(torch.cat([xt, c], 1), use_cl), t).sample
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
                print(f"step {step}/{iters} loss {loss.item():.4f} lr {cur_lr:.2e} {time.time() - t0:.0f}s "
                      f"data-wait {meter.fraction():.0%}", flush=True)
                meter.reset()
            if step % ckpt_every == 0 or step == stop or guard.save_due():
                save()
            if step < stop:
                runtime.pause_if_needed(save, "diffusion")
    logf.close()
    if step >= iters:
        _atomic_save({"unet": unet.state_dict(), "ema": ema.state_dict(), "step": step, "config": cfg}, out / "final.pt")
    return ema


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", required=True, help="final.pt of the lowest-lambda (0.0018) codec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--channels", type=int, nargs=4, default=list(CHANNELS))
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    ap.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    ap.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-list", default=None)
    a = ap.parse_args()
    train_diffusion(a.codec, a.out, iters=a.iters, batch=a.batch, crop=a.crop, lr=a.lr, warmup=a.warmup,
                    ema_decay=a.ema, p_drop=a.p_drop, channels=tuple(a.channels), T=a.T, kind=a.schedule,
                    ckpt_every=a.ckpt_every, workers=a.workers, seed=a.seed, train_list=a.train_list,
                    optimizer=a.optimizer, lr_schedule=a.lr_schedule)


if __name__ == "__main__":
    runtime.run_cli(main)
