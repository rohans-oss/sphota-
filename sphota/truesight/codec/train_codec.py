"""Train the TrueSight VAE codec for one lambda (Step 7).

    python -m truesight.codec.train_codec --lmbda 0.0130 --out experiments/step7_codec/lmbda0.0130
    python -m truesight.codec.train_codec --lmbda 0.0018 --init-from experiments/step7_codec/lmbda0.0130/final.pt \
           --iters 50000 --out experiments/step7_codec/lmbda0.0018          # fine-tune from the lambda=0.0130 model

Adam 1e-4 (x0.1 at 90% of the iterations), a separate Adam (1e-3) for the entropy-bottleneck quantiles,
gradient clipping, a checkpoint every 5k steps and automatic resume from <out>/last.pt.
Writes <out>/log.csv (it, loss, bpp, mse, aux) and <out>/final.pt (weights + config, what eval uses).
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .. import runtime, speed
from ..data import CropDataset, to_float_batch
from .vae_codec import TrueSightCodec, rd_loss

ROOT = Path(__file__).resolve().parents[2]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)                       # a kill during the write can never corrupt the previous checkpoint


def lr_at(it, iters, base_lr, drop_frac=0.9, drop=0.1):
    """Piecewise-constant schedule: base_lr, then base_lr * drop for the last (1 - drop_frac) of training.
    Computed from the iteration number, so resuming needs no scheduler state."""
    return base_lr * (drop if it >= int(iters * drop_frac) else 1.0)


def model_config(net, lmbda):
    return dict(N=net.N, M=net.M, K=net.K, lmbda=lmbda, act=net.act)


def load_codec(path, device="cpu"):
    """Rebuild a codec from a final.pt / last.pt checkpoint. Returns (net in eval mode, config dict)."""
    s = torch.load(path, map_location=device)
    cfg = s["config"]
    net = TrueSightCodec(N=cfg["N"], M=cfg["M"], K=cfg["K"], act=cfg.get("act", "gdn")).to(device)
    net.load_state_dict(_resize_buffers(net, s["net"]))
    return net.eval(), cfg


def _resize_buffers(net, state):
    """EntropyBottleneck CDF buffers (_quantized_cdf, _offset, _cdf_length) are empty until update(); a plain
    load_state_dict would fail on a size mismatch, so resize them to the checkpoint's shapes first."""
    eb = net.entropy_bottleneck
    for name in ("_quantized_cdf", "_offset", "_cdf_length"):
        key = f"entropy_bottleneck.{name}"
        if key in state:
            getattr(eb, name).resize_(state[key].shape)
    return state


def train_codec(lmbda, out, K=3, N=128, M=192, iters=150_000, batch=8, crop=256, lr=1e-4, aux_lr=1e-3, clip=1.0,
                ckpt_every=5000, log_every=100, workers=4, seed=0, train_list=None, init_from=None, device=None,
                stop_at=None, act="gdn", amp=None):
    """Train (or resume) one codec. ``stop_at`` stops early after saving, used to test resume. Returns the net.
    ``act`` is "gdn" or "relu" (ablation A2). ``amp``: fp16 convolutions with fp32 GDN / entropy model / loss (see TrueSightCodec._seq);
    None = decided by TS_CODEC_AMP (default off = pure fp32; "auto" = only after scripts/amp_parity.py passed). Raises ``runtime.Paused`` after saving when the session time budget is used up,
    and writes last.pt before re-raising if anything crashes, so the run always continues from where it stopped."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    speed.setup(device)                                           # cudnn.benchmark (fixed crop size): only changes speed
    net = TrueSightCodec(N=N, M=M, K=K, act=act).to(device)
    use_amp = speed.codec_amp_enabled(device) if amp is None else (bool(amp) and device.type == "cuda")
    net.amp, net._amp_skipped = use_amp, 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)         # loss scaling for the fp16 convolutions (no-op when amp is off)
    use_cl = speed.use_channels_last("codec", device, amp=use_amp)  # off by default: pure fp32 has no Tensor-Core use for channels_last
    if use_cl:
        speed.to_channels_last(net.g_a, net.g_s, net.h_a)
    main_params, aux_params = net.split_parameters()
    fused = speed.fused_ok(device)                                 # one fused kernel for the optimiser step: same update rule
    opt = torch.optim.Adam(main_params, lr=lr, fused=fused)
    aux_opt = torch.optim.Adam(aux_params, lr=aux_lr, fused=fused)
    it = 0
    last = out / "last.pt"
    if last.exists():                                              # resume
        s = torch.load(last, map_location=device)
        net.load_state_dict(_resize_buffers(net, s["net"])); opt.load_state_dict(s["opt"])
        aux_opt.load_state_dict(s["aux"]); it = s["it"]
        if use_amp and "scaler" in s:
            scaler.load_state_dict(s["scaler"])
        seed_all(seed + it)
        print(f"[lmbda={lmbda}] resumed from iteration {it}", flush=True)
    elif init_from:                                                # fine-tune: weights only, fresh optimisers
        s = torch.load(init_from, map_location=device)
        if (s["config"]["K"], s["config"]["M"], s["config"]["N"], s["config"].get("act", "gdn")) != (K, M, N, act):
            raise ValueError(f"init-from config {s['config']} does not match K={K}, M={M}, N={N}, act={act}")
        net.load_state_dict(_resize_buffers(net, s["net"]))
        print(f"[lmbda={lmbda}] initialised from {init_from}", flush=True)

    loader = torch.utils.data.DataLoader(CropDataset(train_list or ROOT / "data" / "processed" / "train.txt", crop, raw=True),
                                         batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                                         persistent_workers=workers > 0, **speed.loader_kwargs(workers, device))
    meter = speed.WaitMeter()                                      # how long the GPU waits for data (see the log line)

    def stream():
        while True:
            yield from loader

    batches = stream()
    new_log = not (out / "log.csv").exists()
    logf = open(out / "log.csv", "a", newline="")
    w = csv.writer(logf)
    if new_log:
        w.writerow(["it", "loss", "bpp", "mse", "aux"]); logf.flush()

    net.train()
    t0 = time.time()
    stop = iters if stop_at is None else min(iters, stop_at)
    guard = runtime.guard()

    def save():                                                    # reads the loop's current ``it`` and weights
        state = {"net": net.state_dict(), "opt": opt.state_dict(), "aux": aux_opt.state_dict(), "it": it,
                 "config": model_config(net, lmbda)}
        if use_amp:
            state["scaler"] = scaler.state_dict()
        _atomic_save(state, last)
        guard.saved()

    with runtime.crash_safe(save, f"lmbda={lmbda}", lambda: runtime.finite_params(net)):
        while it < stop:
            for g in opt.param_groups:
                g["lr"] = lr_at(it, iters, lr)
            with meter:
                raw = next(batches)
            x = speed.cl(to_float_batch(raw, device), use_cl)      # uint8 -> float and the random flip, on the GPU
            out_ = net(x)                                          # entropy model stays float32 (no autocast)
            loss, bpp, mse = rd_loss(out_, x, lmbda)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iteration {it}: loss={loss.item()} bpp={bpp.item()}")
            opt.zero_grad(set_to_none=True)
            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.scale(loss).backward()                          # plain loss.backward() when amp is off
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(main_params, clip)
            scaler.step(opt)                                       # skipped by the scaler when a fp16 gradient overflowed
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                net._amp_skipped += 1
            aux = net.aux_loss()
            aux_opt.zero_grad(set_to_none=True)
            aux.backward()
            aux_opt.step()
            it += 1
            if it % log_every == 0:
                w.writerow([it, loss.item(), bpp.item(), mse.item(), aux.item()]); logf.flush()
                print(f"[lmbda={lmbda}] it {it}/{iters} loss {loss.item():.4f} bpp {bpp.item():.4f} "
                      f"psnr~{-10 * np.log10(max(mse.item(), 1e-12)):.2f} aux {aux.item():.1f} {time.time() - t0:.0f}s "
                      f"data-wait {meter.fraction():.0%}" + (f" amp-skipped {net._amp_skipped}" if use_amp else ""), flush=True)
                meter.reset()
            if it % ckpt_every == 0 or it == stop or guard.save_due():
                save()
            if it < stop:
                runtime.pause_if_needed(save, f"lmbda={lmbda}")
    logf.close()
    net.amp = False                                                # the returned network is for eval / coding: always fp32
    if it >= iters:
        _atomic_save({"net": net.state_dict(), "it": it, "config": model_config(net, lmbda)}, out / "final.pt")
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmbda", type=float, required=True, help="0.0018 0.0035 0.0067 0.0130 0.0250 0.0483")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--M", type=int, default=192)
    ap.add_argument("--iters", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--init-from", default=None, help="final.pt of another lambda to fine-tune from")
    ap.add_argument("--act", choices=["gdn", "relu"], default="gdn", help="activation of g_a / g_s (relu = ablation A2)")
    ap.add_argument("--amp", choices=["0", "1", "auto"], default=None, help="fp16 convolutions (fp32 GDN / entropy model / loss); default: $TS_CODEC_AMP, else 0")
    a = ap.parse_args()
    if a.amp is not None:
        os.environ["TS_CODEC_AMP"] = a.amp                        # train_codec(amp=None) reads it (so "auto" works too)
    train_codec(a.lmbda, a.out, K=a.K, N=a.N, M=a.M, iters=a.iters, batch=a.batch, crop=a.crop, lr=a.lr,
                ckpt_every=a.ckpt_every, workers=a.workers, seed=a.seed, train_list=a.train_list, init_from=a.init_from,
                act=a.act)


if __name__ == "__main__":
    runtime.run_cli(main)
