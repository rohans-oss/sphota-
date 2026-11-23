"""Training ablations A7 (linear vs cosine noise schedule) and A8 (Adam vs AdamW, constant vs cosine learning rate).

Each variant is a short conditional-diffusion training run that differs from the reference run (``base``: cosine schedule,
AdamW, cosine LR) in exactly one setting, followed by the same decode on the same images with the same codec, seeds and
sampler. Because the noise schedule changes what the training loss means, the runs are compared on decode quality
(LPIPS, PSNR against the original) and not on the loss.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..codec.bitstream import pad_to_multiple
from ..metrics import lpips_dist, psnr
from .decode import coarse_decode, diffusion_decode

# name -> (ablation id, keyword arguments of train_diffusion that differ from the reference)
ABLATIONS = {
    "base": ("reference", {}),
    "linear_schedule": ("A7", {"kind": "linear"}),
    "adam": ("A8", {"optimizer": "adam"}),
    "constant_lr": ("A8", {"lr_schedule": "constant"}),
}


def train_loss_tail(log_csv, n=10):
    """Mean of the last ``n`` logged training losses of a run (NaN when the log is missing or empty)."""
    p = Path(log_csv)
    if not p.exists() or p.stat().st_size == 0:
        return float("nan")
    df = pd.read_csv(p)
    return float(df["loss"].tail(n).mean()) if len(df) else float("nan")


def _to_tensor(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(im).permute(2, 0, 1)[None]


@torch.no_grad()
def evaluate_unet(unet, sched, codec, files, steps=25, w=1.5, seed=0, device="cpu", max_images=None, crop=None):
    """Per-image LPIPS / PSNR of the coarse decode and of the diffusion decode of ``codec``'s rounded latents.
    ``crop`` centre-crops each image to at most crop x crop pixels (keeps CPU / smoke runs affordable)."""
    rows = []
    for f in list(files)[: max_images]:
        x = _to_tensor(f)
        if crop:
            h, w_ = x.shape[-2:]
            s = min(crop, h, w_)
            y0, x0 = (h - s) // 2, (w_ - s) // 2
            x = x[..., y0:y0 + s, x0:x0 + s]
        x = x.to(device)
        H, W = x.shape[-2:]
        coarse = coarse_decode(codec, pad_to_multiple(x, codec.DOWN)).clamp(0, 1)[..., :H, :W]
        dec = diffusion_decode(unet, sched, coarse, steps=steps, w=w, seed=seed).clamp(0, 1)
        rows.append(dict(image=Path(f).name, lpips_coarse=lpips_dist(x.cpu(), coarse.cpu()),
                         lpips=lpips_dist(x.cpu(), dec.cpu()), psnr_coarse=psnr(x.cpu(), coarse.cpu()),
                         psnr=psnr(x.cpu(), dec.cpu())))
    return pd.DataFrame(rows)


def summarize_ablations(results):
    """results: {name: per-image DataFrame}. One row per run with mean LPIPS / PSNR and the change relative to ``base``."""
    rows = []
    for name, df in results.items():
        rows.append(dict(run=name, ablation=ABLATIONS.get(name, ("", {}))[0], images=len(df),
                         lpips=float(df["lpips"].mean()), psnr=float(df["psnr"].mean()),
                         lpips_coarse=float(df["lpips_coarse"].mean())))
    out = pd.DataFrame(rows)
    if "base" in set(out["run"]):
        ref = out.loc[out.run == "base", "lpips"].iloc[0]
        out["lpips_vs_base_pct"] = (out["lpips"] / ref - 1) * 100
    return out
