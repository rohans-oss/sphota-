"""Speed settings that change how fast a GPU computes, not what it computes (up to float rounding noise).

    cudnn.benchmark   cuDNN times the convolution algorithms once per input shape and keeps the fastest. Training uses fixed
                      shapes, so the one-off search is paid back. On by default on CUDA.            TS_CUDNN_BENCHMARK=0 turns it off
    channels_last     NHWC memory layout; with fp16 autocast the Tensor Cores of a T4 run convolutions faster in it. PyTorch advises
                      pairing it with AMP, so it is on for the diffusion U-Net (fp16) and OFF for the fp32 codec by default.
                                                                                              TS_CHANNELS_LAST = none | unet | all
    fused EMA         one ``torch._foreach`` call instead of a Python loop over every parameter (in train_diffusion.update_ema).

None of these touch the data, the model, the loss, the optimiser or the iteration counts. ``scripts/profile_train.py`` measures each
setting on your GPU so you can keep only what helps (turn one off with the variables above).
"""
import os

import torch

CHANNELS_LAST_MODES = ("none", "unet", "all")


def _flag(name, default):
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip().lower() not in ("0", "false", "off", "no")


def cudnn_benchmark_enabled():
    return _flag("TS_CUDNN_BENCHMARK", True)


def channels_last_mode():
    m = os.environ.get("TS_CHANNELS_LAST", "unet").strip().lower() or "unet"
    if m not in CHANNELS_LAST_MODES:
        raise ValueError(f"TS_CHANNELS_LAST must be one of {CHANNELS_LAST_MODES}, got {m!r}")
    return m


def setup(device):
    """Process-wide settings for a trainer on ``device``. Does nothing on the CPU."""
    device = torch.device(device)
    if device.type == "cuda" and cudnn_benchmark_enabled():
        torch.backends.cudnn.benchmark = True
    return device


def use_channels_last(kind, device, amp=True):
    """Should the ``kind`` model ("unet" or "codec") run in channels_last on this device?"""
    if torch.device(device).type != "cuda":
        return False
    mode = channels_last_mode()
    return (mode == "all") or (mode == "unet" and kind == "unet" and amp)


def to_channels_last(*modules):
    for m in modules:
        m.to(memory_format=torch.channels_last)


def cl(x, on):
    """``x`` in channels_last layout when ``on`` (same values, different strides)."""
    return x.contiguous(memory_format=torch.channels_last) if on else x


# ------------------------------------------------------------------------------------------------ optimiser / loader helpers
def fused_ok(device):
    """Use the fused CUDA implementation of Adam / AdamW (one kernel for all parameters, same update rule)."""
    return torch.device(device).type == "cuda" and _flag("TS_FUSED_OPTIM", True)


def loader_kwargs(workers, device):
    """Extra DataLoader arguments: pinned host memory on CUDA (faster, async copies) and a deeper prefetch queue for workers."""
    kw = {"pin_memory": torch.device(device).type == "cuda"}
    if workers > 0:
        kw["prefetch_factor"] = int(os.environ.get("TS_PREFETCH", "4"))
    return kw


class WaitMeter:
    """Measures how long a training loop sits waiting for the next batch. ``fraction()`` close to 0: the GPU is fed (the loader is not
    the bottleneck); large (above ~0.1-0.2): the run is data-bound and a faster loader would speed it up.

        with meter: batch = next(batches)         # time spent in the block counts as waiting
    """

    def __init__(self):
        import time
        self._time = time.perf_counter
        self.wait = 0.0
        self.t0 = self._time()
        self._t = None

    def __enter__(self):
        self._t = self._time()

    def __exit__(self, *exc):
        self.wait += self._time() - self._t

    def fraction(self):
        total = self._time() - self.t0
        return self.wait / total if total > 0 else 0.0

    def reset(self):
        self.wait, self.t0 = 0.0, self._time()


# ------------------------------------------------------------------------------------------------ switches that need a parity check
def gated_flag(env, file_env, key, default="0"):
    """A speed-up that can change numerics, so it is only used after a parity check passed:
        <env>=1      force on          <env>=0      off
        <env>=auto   on only if the JSON file named by <file_env> has ``key`` true (written by scripts/amp_parity.py /
                     scripts/eval_parity.py, which compare against the fp32 reference)
    ``default`` is the value used when <env> is unset."""
    import json
    v = os.environ.get(env, default).strip().lower()
    if v in ("1", "true", "on", "yes"):
        return True
    if v == "auto":
        path = os.environ.get(file_env)
        try:
            return bool(path) and bool(json.loads(open(path, encoding="utf-8").read()).get(key))
        except (OSError, ValueError):
            return False
    return False


def codec_amp_enabled(device):
    """fp16 convolutions in codec training (TS_CODEC_AMP = 0 | 1 | auto; default 0 = pure fp32 as in the guide)."""
    return torch.device(device).type == "cuda" and gated_flag("TS_CODEC_AMP", "TS_AMP_PARITY_FILE", "amp_ok", "0")


def eval_amp_enabled(device):
    """fp16 autocast for U-Net decoding (TS_EVAL_AMP = 0 | 1 | auto; default auto: only after eval_parity.py passed)."""
    return torch.device(device).type == "cuda" and gated_flag("TS_EVAL_AMP", "TS_EVAL_PARITY_FILE", "amp_ok", "auto")


def eval_batch_cfg_enabled(device):
    """Run the conditional and unconditional U-Net passes as one batch (TS_EVAL_BATCH_CFG = 0 | 1 | auto; default auto)."""
    return torch.device(device).type == "cuda" and gated_flag("TS_EVAL_BATCH_CFG", "TS_EVAL_PARITY_FILE", "batch_ok", "auto")


def cond_amp_enabled(device):
    """fp16 for the FROZEN codec that makes the conditioning image inside diffusion training (TS_COND_AMP = 0 | 1; default 0).
    Off by default: the decoder must see the same coarse image it gets at test time (which is fp32), and fp16 can round a latent
    differently at a bin boundary."""
    return torch.device(device).type == "cuda" and gated_flag("TS_COND_AMP", "TS_AMP_PARITY_FILE", "cond_ok", "0")
