"""Time budget of the evaluation stages: how many U-Net calls each stage makes and what that costs on THIS GPU.

The cost of a decode is (number of DDIM steps) x (2 U-Net calls per step for classifier-free guidance) x (time of one call at the
image size). ``scripts/benchmark_decode.py`` measures the time of one call at the two sizes the pipeline uses and prints the
projection below, so the full run can be planned (or the settings lowered) BEFORE paying for GPU hours. The projection only counts
U-Net calls (guided decodes also differentiate g_a, which adds maybe a third on top); detectors, OCR and metrics are not counted.
"""
import time

import torch

# the settings of scripts/run_full_pipeline.sh (full scale)
DEFAULTS = dict(kodak=24, text=300, face=500, clic_images=102, tune=20, tune_grid=9, hall=100, hall_seeds=3, table=1500,
                crops=2000, steps=25, main_steps=50, study=40)
CFG_CALLS = 2                    # conditional + unconditional pass per step


def stage_plan(s=None):
    """[(stage, decodes, ddim steps, size key)] where size key is 'img' (the capped image window) or 'crop' (256 x 256)."""
    s = {**DEFAULTS, **(s or {})}
    st, ms = s["steps"], s["main_steps"]
    return [
        ("diffusion_eval", s["kodak"] * (5 + 0), st, "img"),                       # CFG sweep: 5 scales x 25 steps
        ("diffusion_eval", s["kodak"], 10 + st + ms, "img"),                       # steps sweep 10 / 25 / 50 at the chosen w
        ("clic_crops", s["crops"], ms, "crop"),
        ("protect", s["text"] + s["face"], st, "img"),
        ("guidance (tune)", s["tune"] * (1 + s["tune_grid"]), ms, "img"),
        ("guidance (eval)", 4 * (s["text"] + s["face"] + s["kodak"] + s["clic_images"]), ms, "img"),   # 4 variants per image
        ("honesty", s["hall"] * s["hall_seeds"], st, "img"),
        ("intelligence", s["table"] * 2, st, "img"),                               # tiny + small use diffusion (light mode: 5 steps)
        ("study_pairs", s["study"] * 2, ms, "img"),
    ]


def project(call_seconds, plan=None, guided_overhead=1.3):
    """Hours per stage. ``call_seconds`` = {'img': seconds per U-Net call at the image window, 'crop': at 256 x 256}.
    Returns [(stage, decodes, steps, hours)] and the total hours. Guided stages are multiplied by ``guided_overhead``."""
    plan = plan or stage_plan()
    rows, total = [], 0.0
    for stage, n, steps, key in plan:
        h = n * steps * CFG_CALLS * call_seconds[key] / 3600.0
        if "guidance" in stage or stage == "study_pairs":
            h *= guided_overhead
        rows.append((stage, n, steps, h)); total += h
    return rows, total


@torch.no_grad()
def time_unet_call(unet, h, w, device, repeats=3, in_channels=6):
    """Mean seconds of one forward pass at h x w (after one warm-up), batch 1."""
    x = torch.randn(1, in_channels, h, w, device=device)
    t = torch.full((1,), 500, device=device, dtype=torch.long)
    unet(x, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        unet(x, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats
