"""Faithfulness guidance (Step 12, the main contribution) and the full Sphoṭa decode.

Idea. The transmitted integers y_hat define a set of images: every image x whose encoder output g_a(x) rounds to y_hat is
consistent with the file. A hallucinating decoder leaves that set. At each DDIM step the clean-image estimate x0_hat is
pushed back toward it by gradient steps on

    L_bin(x) = mean( ReLU( |g_a(x) - y_hat| - 0.5 + m )^2 ),   m = 0.05      (zero inside every rounding bin)
    L_roi(x) = sum( M * (x - x_roi)^2 ) / (3 * sum(M))                           (protected boxes stay at the decoded crops)
    L        = alpha * L_bin + beta * L_roi

Only x0_hat is differentiated - no gradient flows through the U-Net - so the guidance is cheap. It is guidance in
x0-space, related to Diffusion Posterior Sampling (Chung et al. 2023).

Gradient normalisation. L_bin and L_roi are *means*, so their per-pixel gradients shrink as the image or the mask grows
(~1/(number of latents) and ~1/(number of masked pixels)), which would make ``lr`` mean something different for every
image size. With ``normalize=True`` (default) the optimised objective is rescaled so that ``lr`` is in pixel units at any
size: L_bin by (latent positions)/64, which is exactly 1 for the 128x128 training crops (8x8 latent positions), and L_roi
by the number of masked pixels, which turns its gradient into a per-pixel pull: each inner step removes lr*beta/6 of the
remaining error inside the boxes (8.3% for lr = 0.05, beta = 10; the 1/2 is d x01 / d x for x in [-1, 1]). With
``normalize=False`` the step is the literal gradient of the loss above. ``lr`` is a property of the codec as well: it must
be large enough that the bin loss actually moves (see scripts/tune_guidance.py). The values reported in ``history`` are
always the plain means.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..codec import bitstream as bs
from ..protect import roi as roi_mod
from .decode import diffusion_decode

REF_LATENT_POSITIONS = 64          # latent positions (h * w) of a 128 x 128 training crop at 1/16 resolution


class FaithfulGuide:
    """Callable ``guide(x0, t) -> x0`` for ddim_sample. x0 is in [-1, 1], shape (B, 3, H, W), H and W multiples of 64.

    codec     the low-lambda codec (its g_a defines the quantisation bins), in eval mode
    y_hat     transmitted integer latent (B, M, H/16, W/16) of the same (padded) size
    roi_img   (B, 3, H, W) in [0, 1]: decoded protected crops (zeros elsewhere), or None
    roi_mask  (B, 1, H, W): 1 inside the protected boxes, or None
    t_start   guidance is applied only for timesteps t <= t_start (early, very noisy steps run freely)
    max_step  stability safeguard: no pixel (x in [-1, 1]) moves by more than this in one inner step. Without it an lr
              that is too large for the codec's Jacobian makes the bin loss oscillate and blow up (observed in tests);
              with it the same settings converge. None disables it (the literal gradient step).
    history   one tuple per guided step: (t, L_bin before, L_roi before, L_bin after, L_roi after) - plain means
    """

    def __init__(self, codec, y_hat, roi_img=None, roi_mask=None, alpha=1.0, beta=10.0, inner_steps=3, lr=0.05,
                 t_start=700, margin=0.05, normalize=True, track=True, max_step=0.1):
        self.codec, self.y_hat = codec, y_hat
        self.roi_img, self.roi_mask = roi_img, roi_mask
        self.alpha, self.beta, self.k, self.lr = alpha, beta, inner_steps, lr
        self.t_start, self.m, self.normalize, self.track = t_start, margin, normalize, track
        self.max_step = max_step                  # safeguard: no pixel moves more than this per inner step (None = off)
        self.history = []

    # ------------------------------------------------------------------------------------ losses
    def _parts(self, x01):
        """Per-sample (L_bin, L_roi) means, each of shape (B,), plus the per-sample mask pixel counts."""
        y = self.codec.g_a(x01)
        l_bin = F.relu((y - self.y_hat).abs() - 0.5 + self.m).pow(2).mean(dim=(1, 2, 3))
        if self.roi_mask is not None:
            s = self.roi_mask.sum(dim=(1, 2, 3))
            l_roi = (self.roi_mask * (x01 - self.roi_img) ** 2).sum(dim=(1, 2, 3)) / (3 * s.clamp(min=1))
        else:
            s, l_roi = torch.zeros_like(l_bin), torch.zeros_like(l_bin)
        return l_bin, l_roi, s

    def loss(self, x01):
        """(total loss to minimise, mean L_bin as float, mean L_roi as float). ``x01`` in [0, 1]."""
        l_bin, l_roi, s = self._parts(x01)
        h, w = x01.shape[-2] // 16, x01.shape[-1] // 16
        s_bin = (h * w) / REF_LATENT_POSITIONS if self.normalize else 1.0
        s_roi = s if self.normalize else 1.0
        total = (self.alpha * s_bin * l_bin + self.beta * s_roi * l_roi).sum()
        return total, l_bin.mean().item(), l_roi.mean().item()

    @torch.no_grad()
    def _plain(self, x01):
        l_bin, l_roi, _ = self._parts(x01)
        return l_bin.mean().item(), l_roi.mean().item()

    # ------------------------------------------------------------------------------------ the hook
    def __call__(self, x0, t):
        if t > self.t_start or (self.alpha == 0 and self.roi_mask is None):
            return x0                                        # let the early, very noisy steps run freely
        if self.k == 0:                                      # measure-only: record the losses, change nothing
            lb, lro = self._plain((x0 + 1) / 2)
            self.history.append((t, lb, lro, lb, lro))
            return x0
        x = x0.detach().clone().requires_grad_(True)
        first = None
        with torch.enable_grad():
            for _ in range(self.k):
                total, lb, lro = self.loss((x + 1) / 2)
                if first is None:
                    first = (lb, lro)
                g, = torch.autograd.grad(total, x)
                step = self.lr * g
                if self.max_step is not None:                # an over-strong lr would otherwise overshoot and diverge
                    step = step.clamp(-self.max_step, self.max_step)
                x = (x - step).detach().clamp(-1, 1).requires_grad_(True)
        x = x.detach()
        after = self._plain((x + 1) / 2) if self.track else (float("nan"), float("nan"))
        self.history.append((t, first[0], first[1], after[0], after[1]))
        return x


@torch.no_grad()
def consistency_rate(codec, x01, y_hat, pad=True):
    """Fraction of latent elements whose re-encoding g_a(x) falls inside the transmitted rounding bin |g_a(x) - y_hat| <= 0.5.
    ``x01`` (B, 3, H, W) in [0, 1]; it is edge-padded to a multiple of 64 first (like the encoder did) unless pad=False."""
    if pad:
        x01 = bs.pad_to_multiple(x01, codec.DOWN)
    y = codec.g_a(x01.to(next(codec.parameters()).device))
    return ((y - y_hat.to(y.device)).abs() <= 0.5).float().mean().item()


def pad_to_size(t, H, W):
    """Zero-pad (B, C, h, w) on the bottom / right to (B, C, H, W)."""
    return F.pad(t, (0, W - t.shape[-1], 0, H - t.shape[-2]))


@dataclass
class SphotaOutput:
    """Result of sphota_decode. ``x`` is the final image; ``x_diffusion`` is the diffusion output before the paste."""
    x: torch.Tensor
    x_diffusion: torch.Tensor
    coarse: torch.Tensor
    guide: object = None
    prot: object = None
    boxes: list = field(default_factory=list)


def sphota_decode(codec_lo, codec_hi, unet, sched, blob=None, prot=None, w=1.5, steps=50, seed=0, guided=True,
                  use_roi=True, feather=roi_mod.FEATHER, tile=None, track_only=False, prefix=None, **guide_kw):
    """The full Sphoṭa decoder: decompress the file -> coarse image (bit-exact latent) -> (guided) DDIM with
    classifier-free guidance -> feathered paste of the protected regions.

    blob       the .sphota bytes or path (or pass an already decoded ``prot`` to decode several variants cheaply)
    guided     apply the bin-consistency (and ROI-lock) guidance                      -> ablation A4
    use_roi    use the protected boxes (ROI-lock guidance and final paste)           -> ablation A5
    guided=False, use_roi=False is plain diffusion decoding from the same bits.
    track_only  with guided=False: attach a measure-only guide (no modification) so ``output.guide.history`` holds the
                L_bin of the unguided decode at every step, for the "L_bin along the denoising steps" figure.
    prefix      a sample.PrefixCache of this file: the variants (plain / ROI / guided, different guidance settings) share the denoising
                steps before their guidance starts, so they are computed once (same result as decoding from scratch). Ignored for
                ``track_only`` decodes, which need every step.
    ``guide_kw`` goes to FaithfulGuide (alpha, beta, inner_steps, lr, t_start, margin, normalize). Returns SphotaOutput.
    """
    if prot is None:
        prot = roi_mod.decode_protected(codec_lo, codec_hi, blob, feather)
    dev = prot.coarse.device
    have_roi = bool(use_roi and prot.mask.sum() > 0)
    guide = None
    if track_only and not guided:
        guided, guide_kw = True, dict(guide_kw, inner_steps=0, t_start=10 ** 6)
    if guided:
        H, W = prot.coarse.shape[-2:]
        Hp, Wp = -(-H // codec_lo.DOWN) * codec_lo.DOWN, -(-W // codec_lo.DOWN) * codec_lo.DOWN
        canvas = pad_to_size(prot.canvas, Hp, Wp).to(dev) if have_roi else None
        mask = pad_to_size(prot.mask, Hp, Wp).to(dev) if have_roi else None
        guide = FaithfulGuide(codec_lo, prot.y_hat.to(dev), canvas, mask, **guide_kw)
    x01 = diffusion_decode(unet, sched, prot.coarse, steps=steps, w=w, seed=seed, guide=guide, tile=tile,
                           prefix=None if track_only else prefix)
    x = roi_mod.paste(x01, prot.canvas.to(dev), prot.soft_mask.to(dev)) if have_roi else x01
    return SphotaOutput(x=x.clamp(0, 1), x_diffusion=x01, coarse=prot.coarse, guide=guide, prot=prot,
                        boxes=prot.boxes if have_roi else [])
