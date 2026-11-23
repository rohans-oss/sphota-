"""Noise schedules and the forward process q(x_t | x_0) (Step 9). Shared by the warm-up DDPM and the conditional decoder.

Forward process:   q(x_t | x_0) = N( sqrt(abar_t) x_0 , (1 - abar_t) I ),  abar_t = prod_{s<=t} (1 - beta_s)
                   x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I)

Timesteps are 0-based: t = 0 is the *least* noisy step (abar_0 ~ 1) and t = T-1 the noisiest (abar_{T-1} ~ 0).
Tensors may have any shape (B, ...) with a (B,) integer timestep vector: images (B,3,H,W) and flat data (B,D) both work.
"""
import math

import torch

KINDS = ("linear", "cosine")


def make_betas(T=1000, kind="cosine", beta_start=1e-4, beta_end=0.02, s=0.008, max_beta=0.999):
    """Per-step noise variances beta_1..beta_T as a float64 tensor of length T.

    linear: evenly spaced from beta_start to beta_end (Ho et al. 2020).
    cosine: abar follows cos^2((t/T + s) / (1 + s) * pi/2), normalised to abar_0 = 1 (Nichol & Dhariwal 2021);
            beta_t = 1 - abar_t / abar_{t-1}, clipped at ``max_beta`` so the last steps stay invertible.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown schedule kind {kind!r}; choose one of {KINDS}")
    if T < 2:
        raise ValueError("T must be at least 2")
    if kind == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    return (1 - ab[1:] / ab[:-1]).clamp(max=max_beta)


class Schedule:
    """Holds betas, alphas and alpha-bar (``ab``) and implements the forward process and the x0 / eps conversions."""

    def __init__(self, T=1000, kind="cosine", device="cpu"):
        betas = make_betas(T, kind)
        self.T, self.kind = T, kind
        # cumulative product in float64 (accurate for T = 1000), stored in float32 for the network code
        self.betas = betas.float().to(device)
        self.alphas = (1 - betas).float().to(device)
        self.ab = torch.cumprod(1 - betas, 0).float().to(device)         # alpha-bar
        self.device = torch.device(device)

    def to(self, device):
        for k in ("betas", "alphas", "ab"):
            setattr(self, k, getattr(self, k).to(device))
        self.device = torch.device(device)
        return self

    # ---------------------------------------------------------------- helpers
    def _gather(self, v, t, x):
        """v[t] reshaped to (B, 1, 1, ...) so it broadcasts against x of shape (B, ...)."""
        t = torch.as_tensor(t, device=v.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.numel() and (int(t.min()) < 0 or int(t.max()) >= self.T):
            raise IndexError(f"timestep out of range [0, {self.T - 1}]")
        return v[t].view(-1, *([1] * (x.ndim - 1))).to(x.dtype)

    def signal_scale(self, t, x):
        """sqrt(abar_t), broadcastable against x."""
        return self._gather(self.ab, t, x).sqrt()

    def noise_scale(self, t, x):
        """sqrt(1 - abar_t), broadcastable against x."""
        return (1 - self._gather(self.ab, t, x)).sqrt()

    # ---------------------------------------------------------------- forward process
    def q_sample(self, x0, t, eps):
        """x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
        return self.signal_scale(t, x0) * x0 + self.noise_scale(t, x0) * eps

    # ---------------------------------------------------------------- x0 <-> eps
    def predict_x0(self, xt, t, eps):
        """x0_hat = (x_t - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t)  (inverts q_sample for the given eps)."""
        return (xt - self.noise_scale(t, xt) * eps) / self.signal_scale(t, xt)

    def predict_eps(self, xt, t, x0):
        """eps_hat = (x_t - sqrt(abar_t) x_0) / sqrt(1 - abar_t)  (inverts q_sample for the given x0)."""
        return (xt - self.signal_scale(t, xt) * x0) / self.noise_scale(t, xt)

    def snr(self, t):
        """Signal-to-noise ratio abar_t / (1 - abar_t) at (a tensor of) timesteps."""
        ab = self.ab[torch.as_tensor(t, device=self.ab.device, dtype=torch.long)]
        return ab / (1 - ab)

    # ---------------------------------------------------------------- sampling grid
    def ddim_timesteps(self, steps):
        """``steps`` evenly spaced timesteps from T-1 down to 0 (descending, strictly decreasing, no repeats)."""
        if not 1 <= steps <= self.T:
            raise ValueError(f"steps must be in [1, {self.T}], got {steps}")
        return torch.linspace(self.T - 1, 0, steps).round().long().tolist()
