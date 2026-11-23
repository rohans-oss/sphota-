"""Measure where training time goes, and which speed setting helps on THIS machine (5 minutes on a Kaggle GPU).

    python scripts/profile_train.py                    # loader + GPU step times, guide-size models
    python scripts/profile_train.py --loader-only      # only the data loader (works without a GPU)
    python scripts/profile_train.py --steps 30 --diffusion-batch 32 --codec-batch 8

Part 1, data loader: samples per second of the real DataLoader (CropDataset) without and with the decoded-image cache
(scripts/build_cache.py), for the worker count you give. Compare with part 2: if the loader delivers fewer samples per second than the
GPU consumes, the GPU waits and the cache is what speeds training up.
Part 2, GPU step (synthetic fixed batch, so the loader is not in the way), milliseconds per optimiser step:
    diffusion  cudnn.benchmark off/on  x  channels_last off/on  (fp16 autocast, like train_diffusion), and the EMA update loop vs fused
    codec      cudnn.benchmark off/on, channels_last off/on  (fp32, like train_codec)
    codec amp  the fp16-convolution hybrid (TS_CODEC_AMP, see scripts/amp_parity.py) next to fp32: the speed-up it could give
    attention  fp16 scaled-dot-product attention at the U-Net's token counts: which backend runs on this GPU and how fast
Verdict: whether the run is DATA-BOUND (the loader delivers fewer samples/s than the GPU consumes) or GPU-bound.
Keep what the numbers support. Turn a setting off with TS_CUDNN_BENCHMARK=0 or TS_CHANNELS_LAST=none (see truesight/speed.py).
"""
import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from truesight import speed  # noqa: E402
from truesight.data import CropDataset  # noqa: E402


def loader_rate(files, crop, batch, workers, batches, cache):
    ds = CropDataset(files, crop, cache=cache)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                                     persistent_workers=workers > 0)
    def stream():
        while True:
            yield from dl

    it = stream()
    next(it)                                                          # worker start-up is not counted
    t = time.perf_counter()
    for _ in range(batches):
        next(it)
    return batch * batches / max(time.perf_counter() - t, 1e-9), ds.cache is not None


def timed(fn, steps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps * 1000


def diffusion_step_ms(bench, cl_on, batch, crop, steps, channels):
    from truesight.diffusion.schedule import Schedule
    from truesight.diffusion.train_diffusion import update_ema
    from truesight.diffusion.unet import make_unet
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = bench
    unet = make_unet(crop, channels).to(dev)
    ema = copy.deepcopy(unet).eval().requires_grad_(False)
    if cl_on:
        speed.to_channels_last(unet, ema)
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    sched = Schedule(1000, "cosine", device=dev)
    x0, c = torch.randn(batch, 3, crop, crop, device=dev), torch.randn(batch, 3, crop, crop, device=dev)

    def step():
        t = torch.randint(0, 1000, (batch,), device=dev)
        eps = torch.randn_like(x0)
        xt = sched.q_sample(x0, t, eps)
        with torch.autocast("cuda", dtype=torch.float16):
            out = unet(speed.cl(torch.cat([xt, c], 1), cl_on), t).sample
        loss = F.mse_loss(out.float(), eps)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        update_ema(ema, unet, 0.999)

    ms = timed(step, steps)
    ema_fused = timed(lambda: update_ema(ema, unet, 0.999), steps)

    def loop():
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), unet.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
    return ms, ema_fused, timed(loop, steps)


def attention_ms(tokens, heads=8, dim=32, steps=10):
    """ms per fp16 SDPA call at ``tokens`` positions for each backend that works on this GPU (None = failed / out of memory)."""
    import torch.nn.functional as Fn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(1, heads, tokens, dim, device="cuda", dtype=torch.float16)
    out = {}
    for name, be in (("efficient", SDPBackend.EFFICIENT_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION), ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel(be):
                out[name] = timed(lambda: Fn.scaled_dot_product_attention(q, q, q), steps)
        except Exception:
            out[name] = None
        torch.cuda.empty_cache()
    return out


def codec_step_ms(bench, cl_on, batch, crop, steps, amp=False):
    from truesight.codec.vae_codec import TrueSightCodec, rd_loss
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = bench
    net = TrueSightCodec().to(dev).train()
    net.amp = amp
    if cl_on:
        speed.to_channels_last(net.g_a, net.g_s, net.h_a)
    main, aux = net.split_parameters()
    opt, aopt = torch.optim.Adam(main, lr=1e-4), torch.optim.Adam(aux, lr=1e-3)
    x = speed.cl(torch.rand(batch, 3, crop, crop, device=dev), cl_on)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def step():
        out = net(x)
        loss, _, _ = rd_loss(out, x, 0.013)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        a = net.aux_loss(); aopt.zero_grad(set_to_none=True); a.backward(); aopt.step()
    return timed(step, steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=str(ROOT / "data" / "processed" / "train.txt"))
    ap.add_argument("--workers", type=int, default=2, help="data-loader workers per trainer (2 when two trainers share a 4-core machine)")
    ap.add_argument("--loader-batches", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--diffusion-batch", type=int, default=32)
    ap.add_argument("--diffusion-crop", type=int, default=128)
    ap.add_argument("--codec-batch", type=int, default=8)
    ap.add_argument("--codec-crop", type=int, default=256)
    ap.add_argument("--loader-only", action="store_true")
    a = ap.parse_args()

    print("== data loader (samples per second) ==")
    rates = {}
    try:
        for crop, batch, name in ((a.diffusion_crop, a.diffusion_batch, "diffusion"), (a.codec_crop, a.codec_batch, "codec")):
            r0, _ = loader_rate(a.list, crop, batch, a.workers, a.loader_batches, cache=False)
            r1, used = loader_rate(a.list, crop, batch, a.workers, a.loader_batches, cache=None)
            rates[name] = r1
            print(f"  {name:9s} crop {crop} x batch {batch}, {a.workers} workers: PNG decode {r0:7.1f}/s"
                  + (f"   cache {r1:7.1f}/s  ({r1 / r0:.1f}x)" if used else "   (no cache built: python scripts/build_cache.py)"))
    except FileNotFoundError as e:
        print(f"  skipped: image not found ({e}); download / restore the data first")
    if a.loader_only or not torch.cuda.is_available():
        if not a.loader_only:
            print("no CUDA GPU: GPU timings skipped")
        return

    from truesight.diffusion.unet import CHANNELS
    print(f"== GPU: {torch.cuda.get_device_name(0)} ==")
    print(f"diffusion step (batch {a.diffusion_batch}, {a.diffusion_crop}px, fp16), ms/step; samples/s the GPU can consume")
    base = None
    for bench in (False, True):
        for cl_on in (False, True):
            ms, ema_f, ema_l = diffusion_step_ms(bench, cl_on, a.diffusion_batch, a.diffusion_crop, a.steps, CHANNELS)
            base = base or ms
            gpu_rate = a.diffusion_batch * 1000 / ms
            print(f"  cudnn.benchmark={int(bench)} channels_last={int(cl_on)}: {ms:7.1f} ms ({gpu_rate:6.1f} samples/s, {base / ms:.2f}x)"
                  f"   EMA update fused {ema_f:.2f} ms vs loop {ema_l:.2f} ms")
    if "diffusion" in rates:
        verdict = "DATA-BOUND: the loader is slower than the GPU, a faster loader would speed training up" if rates["diffusion"] < gpu_rate \
            else "GPU-BOUND: the loader keeps up, the GPU is the limit (no gain from loader work)"
        print(f"  verdict (diffusion): loader {rates['diffusion']:.0f}/s vs GPU {gpu_rate:.0f}/s -> {verdict}")
    print(f"codec step (batch {a.codec_batch}, {a.codec_crop}px, fp32), ms/step")
    base = None
    for bench in (False, True):
        for cl_on in (False, True):
            ms = codec_step_ms(bench, cl_on, a.codec_batch, a.codec_crop, a.steps)
            base = base or ms
            print(f"  cudnn.benchmark={int(bench)} channels_last={int(cl_on)}: {ms:7.1f} ms ({a.codec_batch * 1000 / ms:6.1f} samples/s, {base / ms:.2f}x)")
    fp32_ms = base
    amp_ms = codec_step_ms(True, False, a.codec_batch, a.codec_crop, a.steps, amp=True)
    print(f"  fp16 hybrid (amp=1, fp32 GDN / last layers / entropy model): {amp_ms:7.1f} ms ({a.codec_batch * 1000 / amp_ms:6.1f} samples/s, "
          f"{fp32_ms / amp_ms:.2f}x vs fp32 without benchmark flag). Quality is NOT checked here: run scripts/amp_parity.py")
    if "codec" in rates:
        cg = a.codec_batch * 1000 / amp_ms
        print(f"  verdict (codec, even at the fp16 speed): loader {rates['codec']:.0f}/s vs GPU {cg:.0f}/s -> "
              + ("DATA-BOUND" if rates["codec"] < cg else "GPU-BOUND"))
    print("attention, fp16, one head-batch of 8 heads x 32 dims (ms per call; None = not available here)")
    for tokens in (4096, 16384, 24576):
        print(f"  {tokens:6d} tokens: {attention_ms(tokens)}")
    print("Defaults in truesight/speed.py: cudnn.benchmark on; channels_last on for the fp16 U-Net, off for the fp32 codec. Change them with "
          "TS_CUDNN_BENCHMARK=0 / TS_CHANNELS_LAST=none|unet|all if a line above shows they do not help here.")


if __name__ == "__main__":
    main()
