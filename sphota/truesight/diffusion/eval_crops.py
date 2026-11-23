"""Pure helpers for the CLIC 256 x 256 crop evaluation (section 8.1 of the guide: LPIPS and FID on about 2,000 matched crops).

Kept out of the script so they can be unit-tested without a GPU, a checkpoint or the Inception weights.
"""
import random
from pathlib import Path

import pandas as pd
import torch

from .eval_diffusion import checks, summarize

CROP = 256
METHODS = ("coarse", "diffusion", "jpeg", "webp")
FID_NOTE_MIN_CROPS = 1000        # clean-fid on fewer crops than this is noisy and biased: say so in the summary


def crop_positions(H, W, n, size=CROP, seed=0):
    """Up to ``n`` non-overlapping size x size windows of an H x W image, chosen at random (reproducibly) on the size grid.
    Returns a sorted list of (y, x) top-left corners; fewer than n when the image does not hold n windows."""
    cells = [(r * size, c * size) for r in range(H // size) for c in range(W // size)]
    random.Random(seed).shuffle(cells)
    return sorted(cells[:n])


def plan_crops(sizes, per_image=20, max_crops=2000, size=CROP, seed=0):
    """The full, ordered crop list for a dataset. ``sizes`` is [(H, W), ...]; returns [(image index, y, x), ...].
    Every image contributes at most ``per_image`` crops; the list is cut to ``max_crops`` after interleaving the images, so a
    small cap still spreads over all images instead of using only the first ones."""
    per = [[(i, y, x) for y, x in crop_positions(H, W, per_image, size, seed + i)] for i, (H, W) in enumerate(sizes)]
    plan, k = [], 0
    while len(plan) < max_crops and any(k < len(p) for p in per):
        for p in per:
            if k < len(p) and len(plan) < max_crops:
                plan.append(p[k])
        k += 1
    return plan


def per_sample_bpp(likelihoods, num_px):
    """Estimated bits per pixel of EACH sample of a batch from the codec's likelihood tensors ({name: (B, ...)}). -> (B,)."""
    total = 0
    for lik in likelihoods.values():
        total = total + (-torch.log2(lik.clamp(min=1e-9))).flatten(1).sum(1) / num_px
    return total


def compute_fids(root, methods=METHODS, reference="original", fid_fn=None):
    """FID of every method's crop folder against the folder of the original crops: {method: FID}.
    ``fid_fn(dir_a, dir_b) -> float`` defaults to clean-fid's ``compute_fid(mode='clean')``."""
    root = Path(root)
    if fid_fn is None:
        from cleanfid import fid as cfid

        def fid_fn(a, b):
            return float(cfid.compute_fid(str(a), str(b), mode="clean"))
    return {m: float(fid_fn(root / reference, root / m)) for m in methods if (root / m).exists()}


def build_summary(rows, fid=None, meta=None):
    """rows: per-crop DataFrame with columns lpips_<method> (and bpp). Returns the summary dict written to summary.json:
    mean LPIPS per method, the guide's checks (LPIPS reduction of the diffusion decode vs the coarse decode; FID below JPEG,
    WebP and the coarse decode) and a note when there are too few crops for a meaningful FID."""
    long = pd.concat([pd.DataFrame({"method": m, "bpp": rows["bpp"], "lpips": rows[f"lpips_{m}"]}) for m in METHODS
                      if f"lpips_{m}" in rows], ignore_index=True)
    table = summarize(long)
    out = dict(meta or {})
    out.update(crops=int(len(rows)), mean_bpp=float(rows["bpp"].mean()), table=table.round(4).reset_index().to_dict("records"))
    out["checks"] = checks(table, fid)
    if fid:
        out["fid"] = {k: float(v) for k, v in fid.items()}
        if len(rows) < FID_NOTE_MIN_CROPS:
            out["fid_note"] = f"computed on only {len(rows)} crops (the guide uses about 2,000): indicative, not a reliable absolute FID"
    for m in ("jpeg", "webp"):
        if f"reachable_{m}" in rows:
            out[f"{m}_reachable_fraction"] = float(rows[f"reachable_{m}"].mean())
    return out
