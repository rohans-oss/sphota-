"""TrueSight VAE codec (Step 7): GDN autoencoder + hyperprior + discretised Gaussian-mixture entropy model.

    x --g_a--> y --h_a--> z --EntropyBottleneck--> z_hat --h_s--> (w, mu, sigma) per latent element
    y_hat = round(y) is coded with p(y_hat) = sum_k w_k [Phi((y_hat+.5-mu_k)/s_k) - Phi((y_hat-.5-mu_k)/s_k)]
    x_hat = g_s(y_hat)

Rate R = -sum log2 p(y_hat) - sum log2 p(z_hat) (bits an ideal arithmetic coder needs), so minimising the rate is
maximum-likelihood estimation of the entropy model. Loss = bpp + lambda * 255^2 * MSE (a weighted ELBO).
Input height / width must be multiples of 64 (the hyper-encoder downsamples by another factor of 4).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

from .layers import conv, deconv

SCALE_MIN = 0.11          # smallest allowed sigma: keeps the likelihood of a bin finite and the gradient bounded
LOG_SCALE_MAX = 10.0      # exp(10) ~ 2e4 is far beyond any real latent range; guards against fp overflow -> inf/NaN
LAMBDAS = (0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483)   # the six rate-distortion points (Section 7, Step 7)


class GMMConditional(nn.Module):
    """Discretised Gaussian-mixture likelihood for integer-valued (or noisy) latents."""

    def __init__(self, K=3):
        super().__init__()
        self.K = K

    def likelihood(self, y, w_logits, mu, sigma):
        """y: (B,C,H,W); w_logits, mu, sigma: (B,K,C,H,W). Returns p(y) of shape (B,C,H,W), floored at 1e-9."""
        w = F.softmax(w_logits, dim=1)
        sigma = sigma.clamp(min=SCALE_MIN)
        n = torch.distributions.Normal(mu, sigma)
        yk = y.unsqueeze(1)
        p = n.cdf(yk + 0.5) - n.cdf(yk - 0.5)
        return (w * p).sum(dim=1).clamp(min=1e-9)


class TrueSightCodec(nn.Module):
    """N = hidden width, M = latent channels (192 in the guide), K = number of mixture components."""

    DOWN = 64             # required divisibility of H and W

    def __init__(self, N=128, M=192, K=3, act="gdn"):
        """``act`` is the activation of g_a / g_s: "gdn" (the guide's choice) or "relu" (ablation A2)."""
        super().__init__()
        if act not in ("gdn", "relu"):
            raise ValueError(f"act must be 'gdn' or 'relu', got {act!r}")
        self.N, self.M, self.K, self.act = N, M, K, act
        fwd = (lambda: GDN(N)) if act == "gdn" else (lambda: nn.ReLU())
        inv = (lambda: GDN(N, inverse=True)) if act == "gdn" else (lambda: nn.ReLU())
        self.g_a = nn.Sequential(conv(3, N), fwd(), conv(N, N), fwd(),
                                 conv(N, N), fwd(), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), inv(), deconv(N, N), inv(),
                                 deconv(N, N), inv(), deconv(N, 3))
        self.h_a = nn.Sequential(conv(M, N, 3, 1), nn.LeakyReLU(inplace=True),
                                 conv(N, N), nn.LeakyReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, M), nn.LeakyReLU(inplace=True),
                                 deconv(M, M * 3 // 2), nn.LeakyReLU(inplace=True),
                                 conv(M * 3 // 2, 3 * K * M, 3, 1))
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gmm = GMMConditional(K)
        self.amp = False          # mixed precision for TRAINING only (see ``_seq``); set by train_codec, never on at eval / coding

    def _seq(self, seq, x, amp=None, fp32_last=True):
        """Run a Sequential, optionally with mixed precision.

        amp off (default): exactly ``seq(x)``. amp on (CUDA): the plain convolutions run in fp16 under autocast (the Tensor-Core
        work), everything numerically delicate stays fp32: GDN / IGDN / the activations (they divide and square), and, with
        ``fp32_last``, the LAST layer of the stack. That last layer is what produces the latent y that is rounded, the
        hyper-latent z, the mixture parameters (w, mu, sigma) and the reconstruction, so these keep fp32 precision: a fp16 error
        in y would shift which integer it rounds to, a fp16 error in mu / sigma would shift every likelihood. The entropy model,
        the likelihoods, the rate sum, the MSE and the auxiliary loss are all outside these stacks and stay fp32 as well."""
        amp = self.amp if amp is None else amp
        if not amp or not x.is_cuda:
            return seq(x)
        layers = list(seq)
        for i, layer in enumerate(layers):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)) and not (fp32_last and i == len(layers) - 1):
                with torch.autocast("cuda", dtype=torch.float16):
                    x = layer(x)
                x = x.float()
            else:
                x = layer(x.float())
        return x

    def gmm_params(self, z_hat, shape):
        """Mixture parameters for a latent of ``shape`` = (B, M, H, W): (w_logits, mu, sigma), each (B,K,M,H,W)."""
        B, _, H, W = shape
        p = self._seq(self.h_s, z_hat).reshape(B, 3, self.K, self.M, H, W)
        return p[:, 0], p[:, 1], torch.exp(p[:, 2].clamp(max=LOG_SCALE_MAX))

    def forward(self, x):
        if x.shape[-1] % self.DOWN or x.shape[-2] % self.DOWN:
            raise ValueError(f"H and W must be multiples of {self.DOWN}, got {tuple(x.shape[-2:])}")
        y = self._seq(self.g_a, x)
        z = self._seq(self.h_a, y)
        z_hat, z_lik = self.entropy_bottleneck(z)                  # noise in train, rounding in eval
        w, mu, sigma = self.gmm_params(z_hat, y.shape)
        if self.training:
            y_rate = y + torch.empty_like(y).uniform_(-0.5, 0.5)   # uniform noise: differentiable stand-in for rounding
            y_dec = y + (torch.round(y) - y).detach()              # straight-through rounding for the decoder
        else:
            y_rate = y_dec = torch.round(y)
        y_lik = self.gmm.likelihood(y_rate, w, mu, sigma)
        x_hat = self._seq(self.g_s, y_dec)
        return {"x_hat": x_hat, "y": y, "y_hat": y_dec, "likelihoods": {"y": y_lik, "z": z_lik}}

    # ------------------------------------------------------------------ real bitstreams (Step 8)
    def prepare_for_coding(self):
        """Call once after loading a checkpoint, before compress / decompress.

        Builds the CDF tables of the factorised z-model (``entropy_bottleneck.update(force=True)``) and a float32 CPU
        copy of h_s: the mixture parameters, and therefore the arithmetic-coder CDFs, are computed on the CPU in float32
        on both the encoder and the decoder side so that both get bit-identical tables."""
        self.entropy_bottleneck.update(force=True)
        # kept in a dict so nn.Module does not register it (it must stay on the CPU when the net moves to a GPU)
        self._coding = {"h_s": copy.deepcopy(self.h_s).float().cpu().eval()}
        return self

    def _coding_params(self, z_hat, shape):
        """(w_logits, mu, sigma) for the coder, computed on the CPU in float32 (see prepare_for_coding)."""
        B, _, H, W = shape
        p = self._coding["h_s"](z_hat.detach().float().cpu()).view(B, 3, self.K, self.M, H, W)
        return p[:, 0], p[:, 1], torch.exp(p[:, 2].clamp(max=LOG_SCALE_MAX))

    @torch.no_grad()
    def compress(self, x):
        """x: (1, 3, H, W) in [0, 1], H and W multiples of 64. Returns a dict with the two byte streams
        (``z``: hyper-latent, ``y``: latent), their shapes, the transmitted integer latent ``y_hat``
        (= clamp(round(g_a(x)), -L, L)) and ``n_clamped``, the number of latent values that exceeded +-L."""
        from .entropy_gmm import L, encode_y
        if x.shape[0] != 1:
            raise ValueError("compress handles one image at a time (batch size must be 1)")
        if x.shape[-1] % self.DOWN or x.shape[-2] % self.DOWN:
            raise ValueError(f"H and W must be multiples of {self.DOWN}, got {tuple(x.shape[-2:])}")
        if "_coding" not in self.__dict__:
            self.prepare_for_coding()
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])     # the decoder will see exactly this
        w, mu, sigma = self._coding_params(z_hat, y.shape)
        y_hat = torch.round(y)
        n_clamped = int((y_hat.abs() > L).sum())
        y_hat = y_hat.clamp(-L, L)
        return {"z": z_strings[0], "y": encode_y(y_hat, w, mu, sigma), "z_shape": tuple(z.shape[-2:]),
                "y_shape": tuple(y.shape), "y_hat": y_hat.cpu(), "n_clamped": n_clamped}

    @torch.no_grad()
    def decompress(self, d):
        """Inverse of compress (uses only d['z'], d['y'], d['z_shape'], d['y_shape']). Returns x_hat in [0, 1] and the
        decoded integer latent y_hat."""
        from .entropy_gmm import decode_y
        if "_coding" not in self.__dict__:
            self.prepare_for_coding()
        z_hat = self.entropy_bottleneck.decompress([d["z"]], tuple(d["z_shape"]))
        w, mu, sigma = self._coding_params(z_hat, tuple(d["y_shape"]))
        y_hat = decode_y(d["y"], w, mu, sigma)
        dev = next(self.parameters()).device
        return {"x_hat": self.g_s(y_hat.to(dev)).clamp(0, 1), "y_hat": y_hat}

    def aux_loss(self):
        """Loss of the entropy-bottleneck quantiles; optimised separately from the main loss."""
        return self.entropy_bottleneck.loss()

    def split_parameters(self):
        """(main_params, aux_params): the aux optimiser owns only the entropy-bottleneck quantiles."""
        aux = [p for n, p in self.named_parameters() if n.endswith(".quantiles")]
        main = [p for n, p in self.named_parameters() if not n.endswith(".quantiles")]
        return main, aux


def bpp_terms(likelihoods, num_px):
    """Bits per pixel contributed by each likelihood tensor: {name: bpp}; their sum is the total rate."""
    return {k: torch.log(l).sum() / (-math.log(2) * num_px) for k, l in likelihoods.items()}


def rd_loss(out, x, lmbda):
    """Rate-distortion loss: returns (loss, bpp, mse). loss = bpp + lambda * 255^2 * MSE."""
    B, _, H, W = x.shape
    bpp = sum(bpp_terms(out["likelihoods"], B * H * W).values())
    mse = F.mse_loss(out["x_hat"], x)
    return lmbda * 255 ** 2 * mse + bpp, bpp, mse
