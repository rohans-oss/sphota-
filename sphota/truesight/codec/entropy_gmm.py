"""Arithmetic coding of the latent y with the Gaussian-mixture entropy model (Step 8).

The CDF of every latent element over the integer symbols [-L, L] is built from the mixture parameters predicted by the
hyper-decoder, then torchac (an arithmetic coder that accepts any CDF) writes / reads the bytes.

Determinism: the encoder and decoder must compute *identical* CDFs. This module therefore builds them on the CPU in
float32, from parameters that the codec also computes on the CPU in float32 (TrueSightCodec.prepare_for_coding). Float
results of the same network can differ between GPU models or between CPU and GPU, and a one-bit difference in a CDF
makes the arithmetic decoder return garbage. This is a known engineering issue of learned codecs.

torchac compiles a small C++ extension the first time it is imported, so it is imported lazily here: the rest of the
project works without a compiler. On Windows the compiler must be on the PATH (run from an "x64 Native Tools" prompt
or after vcvars64.bat).
"""
import torch
import torch.nn.functional as F

from .vae_codec import SCALE_MIN

L = 60                    # symbols are clamped to [-L, L]; the codec reports how many latent values were clamped
CHUNK = 16                # channels per chunk while building the CDF (bounds the peak memory of the K x 2L+2 tensor)


def torchac_available():
    """True if torchac can be imported (compiler + ninja present). Never raises."""
    try:
        import torchac  # noqa: F401
        return True
    except Exception:
        return False


def _torchac():
    try:
        import torchac
    except Exception as e:                                      # CalledProcessError('where cl'), ImportError, ...
        raise RuntimeError("torchac could not be imported. It compiles a C++ extension on first use: install ninja "
                           "and a C++ compiler (on Windows run from an 'x64 Native Tools' prompt / vcvars64.bat). "
                           f"Original error: {type(e).__name__}: {e}") from e
    return torchac


def gmm_cdf(w_logits, mu, sigma, L=L, chunk=CHUNK):
    """CDF at the 2L+2 bin edges of each element. Inputs (B,K,C,H,W); output (B,C,H,W,2L+2), float32 on the CPU.

    Edge j is at (j - L - 0.5), so symbol s = y + L in [0, 2L] owns the interval [cdf[s], cdf[s+1]]. Each element's CDF
    is renormalised to run exactly from 0 to 1 (mass outside [-L-0.5, L+0.5] is folded into the end symbols' range).
    """
    w_logits, mu, sigma = (t.detach().to(device="cpu", dtype=torch.float32) for t in (w_logits, mu, sigma))
    B, K, C, H, W = mu.shape
    edges = torch.arange(-L, L + 2, dtype=torch.float32) - 0.5
    out = torch.empty(B, C, H, W, 2 * L + 2, dtype=torch.float32)
    for c0 in range(0, C, chunk):
        s = slice(c0, c0 + chunk)
        w = F.softmax(w_logits[:, :, s], dim=1).unsqueeze(-1)                       # (B,K,c,H,W,1)
        n = torch.distributions.Normal(mu[:, :, s].unsqueeze(-1), sigma[:, :, s].clamp(min=SCALE_MIN).unsqueeze(-1))
        out[:, s] = (w * n.cdf(edges)).sum(dim=1)
    lo, hi = out[..., :1], out[..., -1:]
    out = ((out - lo) / (hi - lo).clamp(min=1e-9)).clamp_(0.0, 1.0)
    out[..., -1] = 1.0                                                              # exact end point for every element
    return torch.cummax(out, dim=-1).values                                         # monotone even under rounding noise


def to_symbols(y_hat, L=L):
    """Integer latents -> unsigned symbols in [0, 2L] (int16, CPU). Values outside [-L, L] are clamped."""
    return (y_hat.round().clamp(-L, L) + L).to(torch.int16).cpu()


def encode_y(y_hat, w_logits, mu, sigma, L=L):
    """Arithmetic-code the integer latent y_hat (B,C,H,W) with the GMM CDFs. Returns bytes."""
    ta = _torchac()
    return ta.encode_float_cdf(gmm_cdf(w_logits, mu, sigma, L), to_symbols(y_hat, L), check_input_bounds=True)


def decode_y(byte_stream, w_logits, mu, sigma, L=L):
    """Inverse of encode_y: returns the float latent (B,C,H,W) on the CPU."""
    ta = _torchac()
    return ta.decode_float_cdf(gmm_cdf(w_logits, mu, sigma, L), byte_stream).float() - L
