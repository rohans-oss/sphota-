"""DDIM sampler (the backward process) with classifier-free guidance and a guidance hook (Step 9).

At every step the network's noise estimate eps_hat gives a clean-image estimate
    x0_hat = (x_t - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t)
and the next, less noisy sample is
    x_{t'} = sqrt(abar_t') x0_hat + sqrt(1 - abar_t' - sigma^2) eps_hat + sigma z,
    sigma = eta * sqrt((1 - abar_t') / (1 - abar_t)) * sqrt(1 - abar_t / abar_t').
With eta = 0 (default) the process is deterministic: the same seed always gives the same image, which the decoder
needs so that decoding a file is reproducible. 25-50 steps replace the 1000 of ancestral DDPM sampling.

Two entry points share one loop:
    ddim_sample_eps(eps_fn, sched, shape, ...)   generic: eps_fn(x_t, t) -> eps_hat (unconditional DDPM, toy models, tests)
    ddim_sample(unet, sched, c, ...)             conditional decoder with classifier-free guidance, c = condition image
``unet(x, t)`` must return an object with a ``.sample`` tensor (the diffusers UNet2DModel convention).
Samples live in [-1, 1].
"""
import torch

from .. import speed


class PrefixCache:
    """The denoising path that several decodes of the SAME file share, so it is computed once.

    A decode is deterministic (DDIM, eta = 0): same condition, seed, step count and CFG scale give the same x_t at every step. A
    guided decode only differs from an unguided one from the first step with t <= guide.t_start on (the guide returns x0 unchanged
    before that), and decodes with different guidance settings differ only from their own first guided step. So the unguided /
    ROI-only / guided variants of one image share their whole prefix: the first run records the state x_t before each step it
    shares (on the CPU), later runs resume from the last recorded state they are allowed to start from. The result is the same
    as decoding from scratch (up to float noise); only the repeated work is skipped.

    One cache per image. It forgets everything when the decoding settings change (different condition, steps, w, seed, eta)."""

    def __init__(self):
        self.key, self.states = None, {}

    def check(self, key):
        if key != self.key:
            self.key, self.states = key, {}


class FirstStepCache:
    """The first denoising step of a classifier-free-guided decode is the same for every CFG scale w (same x_T, same condition): only
    the mixing eps = e_u + w (e_c - e_u) differs. A sweep over w computes the two network passes of step 1 once."""

    def __init__(self):
        self.key = self.e_c = self.e_u = None


def divergence_index(ts, guide):
    """Index (into the DDIM timestep list ``ts``) of the first step at which ``guide`` changes x0_hat, or None if it never does
    (no guide, a measure-only guide, or a guide that has nothing to enforce). A custom hook without the FaithfulGuide attributes is
    assumed to act from the first step (nothing is shared)."""
    if guide is None:
        return None
    if getattr(guide, "k", 1) == 0 or (getattr(guide, "alpha", 1) == 0 and getattr(guide, "roi_mask", 1) is None):
        return None
    t_start = getattr(guide, "t_start", None)
    if t_start is None:
        return 0
    for i, t in enumerate(ts):
        if t <= t_start:
            return i
    return None


def _generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def ddim_sample_eps(eps_fn, sched, shape, steps=50, eta=0.0, guide=None, seed=0, device=None, x_T=None, clip=True,
                    return_trace=False, start_index=0, record=None, record_until=None):
    """Run the reverse process.

    eps_fn(x_t, t_batch) -> eps_hat      t_batch is a LongTensor of shape (B,)
    guide(x0_hat, t) -> x0_hat           optional hook, t is the int timestep; if it returns a corrected x0_hat, eps_hat is
                                         re-derived from it so the update stays consistent (this is where Step 12 plugs
                                         in). A hook that returns x0_hat unchanged leaves the decode identical to no hook.
    x_T                                  optional starting noise (default: N(0, I) from ``seed``)
    clip                                 clamp x0_hat to [-1, 1] before the update (eps_hat is left as predicted)
    start_index                         resume at this step: ``x_T`` is then the state x_t BEFORE step ``start_index`` (see PrefixCache)
    record / record_until               a dict that receives {step index: x_t before that step (CPU copy)} for every step index
                                        <= record_until (None = all steps) that it does not already hold
    Returns x_0 in [-1, 1] (and, with return_trace, the list of (t, x0_hat) pairs).
    """
    device = torch.device(device) if device is not None else sched.device
    g = _generator(device, seed)
    x = torch.randn(shape, generator=g, device=device) if x_T is None else x_T.to(device).clone()
    ts = sched.ddim_timesteps(steps)
    trace = []
    for i in range(start_index, len(ts)):
        t = ts[i]
        if record is not None and i not in record and (record_until is None or i <= record_until):
            record[i] = x.detach().cpu()
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tt)
        ab = sched.ab[t].to(device)
        ab_prev = sched.ab[t_prev].to(device) if t_prev >= 0 else torch.ones((), device=device)
        x0 = sched.predict_x0(x, tt, eps)
        if clip:
            x0 = x0.clamp(-1, 1)
        if guide is not None:
            x0_guided = guide(x0, t)
            if x0_guided is not x0 and not torch.equal(x0_guided, x0):
                x0 = x0_guided
                eps = sched.predict_eps(x, tt, x0)          # x0 was corrected: re-derive eps from it (as in the guide)
            # a hook that changed nothing (skipped step, measure-only) leaves the trajectory identical to no hook
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt() if t_prev >= 0 else 0.0
        dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        x = ab_prev.sqrt() * x0 + dir_xt
        if eta > 0 and t_prev >= 0:
            x = x + sigma * torch.randn(x.shape, generator=g, device=device)
        if return_trace:
            trace.append((t, x0.clone()))
    return (x, trace) if return_trace else x


@torch.no_grad()
def ddim_sample(unet, sched, c, steps=50, w=1.5, guide=None, seed=0, eta=0.0, x_T=None, amp=None, batch_cfg=None, prefix=None,
                first_step=None):
    """Conditional DDIM with classifier-free guidance: eps = e_u + w (e_c - e_u).

    c: condition image (B, C, H, W) in [-1, 1] (the coarse reconstruction); the null condition is zeros.
    w = 1 -> pure conditional (one network pass per step); w = 0 -> unconditional; larger w -> sharper but riskier.
    The network input is cat([x_t, condition], dim=1), i.e. 2C channels.

    Speed options that do not change the result beyond float noise:
      batch_cfg    run the conditional and unconditional passes as ONE batch of 2 (same maths per sample; better GPU use). Falls
                   back to two passes if the batch does not fit in memory. None = ``speed.eval_batch_cfg_enabled`` (needs the
                   parity check, see scripts/eval_parity.py).
      amp          fp16 autocast for the U-Net (it was trained under fp16 autocast); eps is returned in fp32. None =
                   ``speed.eval_amp_enabled`` (needs the parity check).
      prefix       a PrefixCache: reuse the denoising prefix shared with other decodes of the same file (guided vs unguided ...).
      first_step   a FirstStepCache: reuse the first step across a sweep of CFG scales.
    """
    device = c.device
    use_amp = speed.eval_amp_enabled(device) if amp is None else bool(amp) and device.type == "cuda"
    use_batch = speed.eval_batch_cfg_enabled(device) if batch_cfg is None else bool(batch_cfg)   # explicit True works on any device
    null = torch.zeros_like(c)
    state = {"batch": use_batch, "calls": 0}

    def net(x_in, tt):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = unet(x_in, tt).sample
        return out.float() if use_amp else out

    def both(x, tt, need_c=True, need_u=True):
        """(e_c, e_u): the conditional / unconditional predictions (None for the one not needed)."""
        if need_c and need_u and state["batch"]:
            try:
                out = net(torch.cat([torch.cat([x, c], 1), torch.cat([x, null], 1)], 0), torch.cat([tt, tt], 0))
                e_c, e_u = out.chunk(2)
                return e_c, e_u
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                state["batch"] = False                              # a batch of 2 does not fit: two passes from now on
        e_c = net(torch.cat([x, c], 1), tt) if need_c else None
        e_u = net(torch.cat([x, null], 1), tt) if need_u else None
        return e_c, e_u

    start = 0
    ts = sched.ddim_timesteps(steps)
    record = record_until = None
    if prefix is not None:
        key = (steps, float(w), seed, float(eta), tuple(c.shape), float(c.double().sum()))
        prefix.check(key)
        i0 = divergence_index(ts, guide)
        limit = len(ts) - 1 if i0 is None else i0
        have = [i for i in prefix.states if i <= limit]
        if have and x_T is not None:
            start = max(have)
            x_T = prefix.states[start].to(device)
        record, record_until = prefix.states, i0

    def eps_fn(x, tt):
        first = state["calls"] == 0 and start == 0 and first_step is not None
        state["calls"] += 1
        if first:
            fk = (tuple(x.shape), float(x.double().sum()), float(c.double().sum()), int(tt[0]))
            if first_step.key != fk:
                first_step.key, first_step.e_c, first_step.e_u = fk, None, None
            need_c, need_u = first_step.e_c is None, first_step.e_u is None and w != 1
            if need_c or need_u:
                e_c, e_u = both(x, tt, need_c, need_u)
                first_step.e_c = e_c if e_c is not None else first_step.e_c
                first_step.e_u = e_u if e_u is not None else first_step.e_u
            e_c, e_u = first_step.e_c, first_step.e_u
        elif w == 1:
            e_c, e_u = both(x, tt, True, False)[0], None
        else:
            e_c, e_u = both(x, tt)
        return e_c if w == 1 else e_u + w * (e_c - e_u)

    return ddim_sample_eps(eps_fn, sched, c.shape, steps=steps, eta=eta, guide=guide, seed=seed, device=c.device,
                           x_T=x_T, start_index=start, record=record, record_until=record_until)
