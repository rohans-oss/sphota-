"""Building blocks for the codecs (Step 6 onwards): conv / deconv helpers and the PlainAE baseline."""
import numpy as np
import torch
import torch.nn as nn


def conv(i, o, k=5, s=2):
    """k x k convolution that halves the resolution when s = 2 (needs H, W divisible by s)."""
    return nn.Conv2d(i, o, k, s, k // 2)


def deconv(i, o, k=5, s=2):
    """Transposed convolution that doubles the resolution when s = 2."""
    return nn.ConvTranspose2d(i, o, k, s, k // 2, output_padding=s - 1)


def round_ste(y):
    """Round to integers in the forward pass; identity gradient in the backward pass."""
    return y + (torch.round(y) - y).detach()


class PlainAE(nn.Module):
    """Plain convolutional autoencoder: 4 stride-2 convs -> M-channel bottleneck -> 4 stride-2 deconvs.

    Trained with MSE only, so nothing in the loss limits (or even measures) the bit cost of the latent.
    The bottleneck is rounded with a straight-through estimator, so the same integers are used at train and
    test time. Input / output: (B, 3, H, W) with H, W divisible by 16; the latent is (B, M, H/16, W/16).
    """
    DOWN = 16

    def __init__(self, N=128, M=32):
        super().__init__()
        self.N, self.M = N, M
        self.enc = nn.Sequential(conv(3, N), nn.ReLU(), conv(N, N), nn.ReLU(),
                                 conv(N, N), nn.ReLU(), conv(N, M))
        self.dec = nn.Sequential(deconv(M, N), nn.ReLU(), deconv(N, N), nn.ReLU(),
                                 deconv(N, N), nn.ReLU(), deconv(N, 3))

    def encode(self, x):
        """Real-valued latent y (before rounding)."""
        return self.enc(x)

    def decode(self, y_hat):
        return self.dec(y_hat)

    def forward(self, x):
        return self.dec(round_ste(self.enc(x)))

    @torch.no_grad()
    def reconstruct(self, x):
        """Eval path: returns (reconstruction clamped to [0, 1], integer latent as an int32 tensor)."""
        q = torch.round(self.enc(x))
        return self.dec(q).clamp(0, 1), q.to(torch.int32)


def entropy_bits(symbols):
    """Ideal entropy-coded size in bits = -sum_s count_s * log2(p_s), with p_s the empirical (MLE) frequency.

    ``symbols`` is any integer array / tensor; all elements are pooled into one histogram. This is the size an
    ideal arithmetic coder with a *fixed* histogram would reach (the histogram itself is not charged).
    """
    s = symbols.detach().cpu().numpy() if torch.is_tensor(symbols) else np.asarray(symbols)
    _, counts = np.unique(s.ravel(), return_counts=True)
    p = counts / counts.sum()
    return float(-(counts * np.log2(p)).sum())
