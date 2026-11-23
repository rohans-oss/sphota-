"""Training and evaluation of the PlainAE baseline (Step 6). CLI wrapper: scripts/train_ae.py."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import runtime, speed
from ..data import CropDataset, FullImageDataset
from ..metrics import psnr
from .layers import PlainAE, entropy_bits


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _batches(loader):
    while True:
        yield from loader


def save_ckpt(path, model, opt, it, M, N, seed, log_rows):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it, "M": M, "N": N, "seed": seed,
                "log": log_rows}, tmp)
    tmp.replace(path)                                    # atomic: a kill mid-write never corrupts the checkpoint


def train_ae(train_list, out_dir, M=32, N=128, iters=30000, batch=16, crop=256, lr=1e-4, seed=0,
             ckpt_every=500, log_every=50, workers=0, device=None, stop_at=None):
    """Train a PlainAE with MSE only. Resumes from ``out_dir/ckpt.pt`` if it exists.

    ``stop_at`` stops early (after saving a checkpoint) at that iteration, used to test resume.
    Returns (model, log_rows). Log rows: dict(iter, mse, psnr_batch, sec).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    speed.setup(device)                                           # cudnn.benchmark: only changes speed
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = PlainAE(N, M).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    it, log = 0, []
    ck = out_dir / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); it, log = s["iter"], s["log"]
        seed_all(seed + it)                              # different crops after a resume, still deterministic
        print(f"[M={M}] resumed from iteration {it}", flush=True)
    loader = torch.utils.data.DataLoader(CropDataset(train_list, crop), batch_size=batch, shuffle=True,
                                         num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    stream = _batches(loader)
    model.train()
    t0, run = time.time(), []
    last = iters if stop_at is None else min(iters, stop_at)
    guard = runtime.guard()

    def save():                                          # reads the loop's current ``it`` and weights
        save_ckpt(ck, model, opt, it, M, N, seed, log)
        guard.saved()

    with runtime.crash_safe(save, f"M={M}", lambda: runtime.finite_params(model)):
        while it < last:
            x = next(stream).to(device)
            loss = F.mse_loss(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            it += 1
            run.append(loss.item())
            if it % log_every == 0 or it == iters:
                mse = float(np.mean(run)); run = []
                log.append(dict(iter=it, mse=mse, psnr_batch=float(10 * np.log10(1 / mse)), sec=round(time.time() - t0, 1)))
                print(f"[M={M}] it {it}/{iters}  mse {mse:.5f}  psnr~{log[-1]['psnr_batch']:.2f} dB  {log[-1]['sec']}s", flush=True)
            if it % ckpt_every == 0 or it == last or guard.save_due():
                save()
            if it < last:
                runtime.pause_if_needed(save, f"M={M}")
    with open(out_dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "mse", "psnr_batch", "sec"]); w.writeheader(); w.writerows(log)
    return model, log


@torch.no_grad()
def evaluate_ae(model, test_list, device=None, max_images=None):
    """Full-image evaluation. Returns dict(psnr, bpp, bits_per_image, latent stats) and a list of recon tensors.

    PSNR is averaged over images (on the unpadded region). Bits = ideal entropy of the rounded latents pooled
    over the whole test set (one histogram), divided by the number of original pixels -> bpp.
    """
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    ds = FullImageDataset(test_list, multiple=PlainAE.DOWN)
    n = len(ds) if max_images is None else min(max_images, len(ds))
    psnrs, latents, pixels, recons = [], [], 0, []
    for i in range(n):
        x, (h, w), _ = ds[i]
        rec, q = model.reconstruct(x[None].to(device))
        rec = rec[..., :h, :w].cpu()
        psnrs.append(psnr(x[None, :, :h, :w], rec))
        latents.append(q.cpu().flatten())
        pixels += h * w
        recons.append(rec[0])
    allq = torch.cat(latents)
    bits = entropy_bits(allq)
    per_img = [entropy_bits(l) for l in latents]
    return dict(psnr=float(np.mean(psnrs)), bpp=bits / pixels, bits_total=bits, n_images=n,
                bits_per_image_mean=float(np.mean(per_img)), latent_min=int(allq.min()), latent_max=int(allq.max()),
                latent_unique=int(torch.unique(allq).numel()), latent_mean_abs=float(allq.abs().float().mean()),
                latent_elems_per_pixel=float(allq.numel() / pixels)), recons
