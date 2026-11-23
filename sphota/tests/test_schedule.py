from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from truesight.diffusion.sample import ddim_sample, ddim_sample_eps, _generator
from truesight.diffusion.schedule import Schedule, make_betas

KINDS = ["linear", "cosine"]


# --------------------------------------------------------------------------- schedule
@pytest.mark.parametrize("kind", KINDS)
def test_alpha_bar_properties(kind):
    s = Schedule(1000, kind)
    assert s.ab.shape == (1000,) and s.betas.shape == (1000,)
    assert s.ab[0] > 0.99                                            # abar_0 ~ 1: the first step adds almost no noise
    assert (s.ab[1:] < s.ab[:-1]).all()                              # strictly decreasing
    assert s.ab[-1] < 0.01                                           # the last step is (almost) pure noise
    assert (s.betas > 0).all() and (s.betas <= 0.999).all()
    assert torch.allclose(s.ab, torch.cumprod(s.alphas, 0), rtol=1e-4)


def test_linear_betas_match_ho_et_al():
    b = make_betas(1000, "linear")
    assert b[0].item() == pytest.approx(1e-4) and b[-1].item() == pytest.approx(0.02)
    assert torch.allclose(b.diff(), torch.full((999,), (0.02 - 1e-4) / 999, dtype=torch.float64))


def test_cosine_abar_follows_the_cosine_formula():
    import math
    T, off = 1000, 0.008
    s = Schedule(T, "cosine")
    f = lambda t: math.cos((t / T + off) / (1 + off) * math.pi / 2) ** 2
    for t in (1, 100, 500, 900):                                     # ab[t-1] is abar after t steps
        assert s.ab[t - 1].item() == pytest.approx(f(t) / f(0), rel=2e-3)


def test_cosine_destroys_the_signal_more_slowly_than_linear():
    lin, cos = Schedule(1000, "linear"), Schedule(1000, "cosine")
    assert (cos.ab[1:900] > lin.ab[1:900]).all()                     # cosine keeps more signal at every step in between
    assert cos.ab[500].item() == pytest.approx(0.49, abs=0.02) and lin.ab[500].item() == pytest.approx(0.078, abs=0.01)
    # linear is already ~pure noise (abar < 0.01) after two thirds of the steps, cosine only near the very end
    assert int((lin.ab < 0.01).nonzero()[0]) < 700 < 900 < int((cos.ab < 0.01).nonzero()[0])


def test_invalid_arguments():
    with pytest.raises(ValueError):
        Schedule(1000, "quadratic")
    with pytest.raises(ValueError):
        Schedule(1, "linear")
    with pytest.raises(ValueError):
        Schedule(100).ddim_timesteps(0)
    with pytest.raises(ValueError):
        Schedule(100).ddim_timesteps(101)
    s = Schedule(100)
    with pytest.raises(IndexError):
        s.q_sample(torch.zeros(1, 3, 4, 4), torch.tensor([100]), torch.zeros(1, 3, 4, 4))


# --------------------------------------------------------------------------- forward process
@pytest.mark.parametrize("kind", KINDS)
def test_q_sample_matches_the_formula(kind):
    s = Schedule(1000, kind)
    g = torch.Generator().manual_seed(0)
    x0, eps = torch.randn(4, 3, 8, 8, generator=g), torch.randn(4, 3, 8, 8, generator=g)
    t = torch.tensor([0, 10, 500, 999])
    xt = s.q_sample(x0, t, eps)
    for i, ti in enumerate(t.tolist()):
        want = s.ab[ti].sqrt() * x0[i] + (1 - s.ab[ti]).sqrt() * eps[i]
        assert torch.allclose(xt[i], want, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_q_sample_at_the_extremes(kind):
    s = Schedule(1000, kind)
    g = torch.Generator().manual_seed(1)
    x0 = torch.rand(64, 3, 32, 32, generator=g) * 2 - 1              # an "image" in [-1, 1], mean ~ 0
    eps = torch.randn(x0.shape, generator=g)
    assert (s.q_sample(x0, torch.zeros(64, dtype=torch.long), eps) - x0).abs().max() < 0.06       # t = 0: still the image
    xT = s.q_sample(x0, torch.full((64,), 999), eps)
    assert abs(xT.mean().item()) < 0.02 and xT.std().item() == pytest.approx(1.0, abs=0.03)      # t = T-1: ~ N(0, 1)


@pytest.mark.parametrize("kind", KINDS)
def test_q_sample_preserves_unit_variance_for_unit_variance_data(kind):
    s = Schedule(1000, kind)
    g = torch.Generator().manual_seed(2)
    x0 = torch.randn(200_000, generator=g)
    for t in (0, 100, 500, 900, 999):                                # var = abar * 1 + (1 - abar) = 1 at every t
        xt = s.q_sample(x0, torch.full((200_000,), t), torch.randn(200_000, generator=g))
        assert xt.var().item() == pytest.approx(1.0, abs=0.02)


def test_works_for_flat_and_image_tensors_and_scalar_t():
    s = Schedule(100)
    assert s.q_sample(torch.zeros(5, 2), torch.arange(5), torch.ones(5, 2)).shape == (5, 2)
    assert s.q_sample(torch.zeros(5, 3, 4, 4), 7, torch.ones(5, 3, 4, 4)).shape == (5, 3, 4, 4)   # scalar t -> whole batch


@pytest.mark.parametrize("kind", KINDS)
def test_x0_and_eps_prediction_invert_the_forward_process(kind):
    s = Schedule(1000, kind)
    g = torch.Generator().manual_seed(3)
    x0, eps = torch.randn(8, 3, 6, 6, generator=g), torch.randn(8, 3, 6, 6, generator=g)
    t = torch.tensor([0, 1, 50, 200, 500, 800, 950, 998])
    xt = s.q_sample(x0, t, eps)
    assert torch.allclose(s.predict_x0(xt, t, eps), x0, atol=2e-3)
    assert torch.allclose(s.predict_eps(xt, t, x0), eps, atol=2e-3)


def test_snr_decreases_with_t():
    s = Schedule(1000, "cosine")
    snr = s.snr(torch.arange(1000))
    assert (snr[1:] < snr[:-1]).all() and snr[0] > 100 and snr[-1] < 0.01


# --------------------------------------------------------------------------- sampling grid
@pytest.mark.parametrize("steps", [1, 2, 10, 25, 50, 1000])
def test_ddim_timesteps_are_descending_unique_and_span_the_range(steps):
    ts = Schedule(1000).ddim_timesteps(steps)
    assert len(ts) == steps and ts[0] == 999 and ts[-1] == 0 if steps > 1 else ts == [999]
    assert all(a > b for a, b in zip(ts, ts[1:]))


# --------------------------------------------------------------------------- DDIM with an exact denoiser
def _exact_eps_fn(sched, m, s):
    """Optimal eps predictor when every element of x0 is N(m, s^2): E[eps | x_t] = sqrt(1-ab)(x_t - sqrt(ab) m) / (ab s^2 + 1 - ab)."""
    def fn(x, tt):
        ab = sched.ab[tt].view(-1, *([1] * (x.ndim - 1)))
        return (1 - ab).sqrt() * (x - ab.sqrt() * m) / (ab * s ** 2 + 1 - ab)
    return fn


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("steps", [10, 50])
def test_ddim_with_exact_denoiser_recovers_a_point_mass(kind, steps):
    sched = Schedule(1000, kind)
    x = ddim_sample_eps(_exact_eps_fn(sched, 0.3, 1e-6), sched, (16, 3, 4, 4), steps=steps, seed=1)
    assert (x - 0.3).abs().max() < 1e-3                              # whatever the starting noise, it lands on the data point


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_ddim_with_exact_denoiser_converges_to_a_gaussian_as_steps_grow(kind, eta):
    """A discrete DDIM / ancestral step with the *exact* denoiser keeps the mean but slightly under-disperses the
    spread (Cauchy-Schwarz); the error must shrink monotonically to zero as the number of steps grows."""
    sched = Schedule(1000, kind)
    fn = _exact_eps_fn(sched, 0.3, 0.15)
    stds = []
    for steps in (10, 50, 250, 1000):
        x = ddim_sample_eps(fn, sched, (40_000, 1), steps=steps, eta=eta, seed=2)
        assert x.mean().item() == pytest.approx(0.3, abs=0.01)                      # the mean is right at every step count
        stds.append(x.std().item())
    assert stds == sorted(stds) and stds[0] < 0.11                                   # coarse sampling visibly under-disperses
    assert stds[-1] == pytest.approx(0.15, rel=0.03)                                 # 1000 steps: the right distribution


def test_cosine_needs_fewer_steps_than_linear_for_the_same_accuracy():
    err = {}
    for kind in KINDS:
        sched = Schedule(1000, kind)
        x = ddim_sample_eps(_exact_eps_fn(sched, 0.3, 0.15), sched, (40_000, 1), steps=25, seed=2)
        err[kind] = abs(x.std().item() - 0.15)
    assert err["cosine"] < err["linear"]


def test_ddim_eta_one_is_stochastic_but_reproducible_for_a_fixed_seed():
    sched = Schedule(1000, "cosine")
    fn = _exact_eps_fn(sched, -0.2, 0.2)
    a = ddim_sample_eps(fn, sched, (4_000, 1), steps=100, eta=1.0, seed=3)
    assert torch.equal(a, ddim_sample_eps(fn, sched, (4_000, 1), steps=100, eta=1.0, seed=3))
    assert not torch.equal(a, ddim_sample_eps(fn, sched, (4_000, 1), steps=100, eta=0.0, seed=3))
    assert a.mean().item() == pytest.approx(-0.2, abs=0.02)


def test_ddim_is_deterministic_in_the_seed_and_x_T():
    sched = Schedule(1000)
    fn = _exact_eps_fn(sched, 0.0, 0.5)
    a = ddim_sample_eps(fn, sched, (4, 8), steps=25, seed=5)
    assert torch.equal(a, ddim_sample_eps(fn, sched, (4, 8), steps=25, seed=5))
    assert not torch.equal(a, ddim_sample_eps(fn, sched, (4, 8), steps=25, seed=6))
    xT = torch.randn((4, 8), generator=_generator(torch.device("cpu"), 5))
    assert torch.equal(a, ddim_sample_eps(fn, sched, (4, 8), steps=25, x_T=xT))      # same seed == passing that noise


def test_ddim_output_stays_in_range_and_trace_has_one_entry_per_step():
    sched = Schedule(1000)
    wild = lambda x, tt: x * 10.0                                    # a bad network: huge eps estimates
    x, trace = ddim_sample_eps(wild, sched, (2, 3, 4, 4), steps=7, seed=0, return_trace=True)
    assert [t for t, _ in trace] == sched.ddim_timesteps(7) and all(x0.abs().max() <= 1 for _, x0 in trace)


# --------------------------------------------------------------------------- guidance hook and CFG
def test_guide_hook_is_called_with_descending_t_and_its_output_is_used():
    sched, calls = Schedule(1000), []

    def guide(x0, t):
        calls.append(t)
        return torch.zeros_like(x0)

    out = ddim_sample_eps(lambda x, tt: torch.zeros_like(x), sched, (2, 3), steps=5, guide=guide, seed=0)
    assert calls == sched.ddim_timesteps(5) and torch.count_nonzero(out) == 0          # last x_0 is the guided x0


def test_identity_guide_changes_nothing_when_no_clipping_is_active():
    sched = Schedule(1000)
    fn = _exact_eps_fn(sched, 0.3, 1e-6)                             # x0_hat is always exactly 0.3: no clipping, eps consistent
    a = ddim_sample_eps(fn, sched, (8, 4), steps=20, seed=1)
    b = ddim_sample_eps(fn, sched, (8, 4), steps=20, seed=1, guide=lambda x0, t: x0)
    assert torch.allclose(a, b, atol=1e-5)


def test_a_hook_that_changes_nothing_leaves_the_decode_identical_even_when_clipping_is_active():
    sched = Schedule(1000)
    wild = lambda x, tt: x * 3.0                                      # large eps estimates -> x0_hat is clipped at +-1 often
    plain = ddim_sample_eps(wild, sched, (2, 3, 4, 4), steps=8, seed=0)
    noop = ddim_sample_eps(wild, sched, (2, 3, 4, 4), steps=8, seed=0, guide=lambda x0, t: x0)
    same_values = ddim_sample_eps(wild, sched, (2, 3, 4, 4), steps=8, seed=0, guide=lambda x0, t: x0.clone())
    assert torch.equal(plain, noop) and torch.equal(plain, same_values)
    changed = ddim_sample_eps(wild, sched, (2, 3, 4, 4), steps=8, seed=0, guide=lambda x0, t: x0 * 0.5)
    assert not torch.equal(plain, changed)                            # a real correction still changes the trajectory


class _FakeUNet:
    """eps depends on the condition: e(x, c) = 0.1 x + 0.5 c. Counts forward passes. Input is cat([x_t, c])."""
    def __init__(self, C=3):
        self.calls, self.C = 0, C

    def __call__(self, inp, tt):
        self.calls += 1
        x, c = inp[:, :self.C], inp[:, self.C:]
        return SimpleNamespace(sample=0.1 * x + 0.5 * c)


@pytest.mark.parametrize("w", [0.0, 1.0, 1.5, 3.0])
def test_cfg_combines_conditional_and_unconditional_predictions(w):
    sched, unet = Schedule(1000), _FakeUNet()
    c = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(0)).clamp(-1, 1)
    xT = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(1))
    out = ddim_sample(unet, sched, c, steps=1, w=w, x_T=xT)           # one step: x_0 = clamp(x0_hat)
    e_c, e_u = 0.1 * xT + 0.5 * c, 0.1 * xT
    eps = e_u + w * (e_c - e_u)
    want = sched.predict_x0(xT, torch.full((2,), 999), eps).clamp(-1, 1)
    assert torch.allclose(out, want, atol=1e-5)
    assert unet.calls == (1 if w == 1 else 2)                         # w = 1 needs only the conditional pass


def test_cfg_with_a_neutral_network_leaves_a_deterministic_result():
    sched = Schedule(1000)
    c = torch.zeros(1, 3, 8, 8)
    a = ddim_sample(_FakeUNet(), sched, c, steps=5, w=1.5, seed=4)
    assert a.shape == c.shape and torch.equal(a, ddim_sample(_FakeUNet(), sched, c, steps=5, w=1.5, seed=4))


# --------------------------------------------------------------------------- the shared code learns a real distribution
@pytest.mark.parametrize("kind", KINDS)
def test_toy_ddpm_trained_with_q_sample_samples_the_data_distribution(kind):
    """Train a tiny eps-network on a 2-mode 2-D distribution using Schedule.q_sample, sample it with ddim_sample_eps.
    The same code path as the image DDPM: if forward or backward diffusion were wrong the samples would be blobs of
    noise instead of two tight clusters."""
    torch.manual_seed(0)
    T = 200
    sched = Schedule(T, kind)
    net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    modes = torch.tensor([[-0.6, 0.0], [0.6, 0.0]])

    def batch(n):
        return modes[torch.randint(0, 2, (n,))] + 0.05 * torch.randn(n, 2)

    for _ in range(2500):
        x0 = batch(256)
        t = torch.randint(0, T, (256,))
        eps = torch.randn_like(x0)
        loss = ((net(torch.cat([sched.q_sample(x0, t, eps), (t.float() / T)[:, None]], 1)) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    fn = lambda x, tt: net(torch.cat([x, (tt.float() / T)[:, None]], 1))
    with torch.no_grad():
        s = ddim_sample_eps(fn, sched, (2000, 2), steps=25, seed=0)
    d = torch.cdist(s, modes).min(1).values
    left = (s[:, 0] < 0).float().mean().item()
    assert (d < 0.2).float().mean() > 0.9, "samples are not concentrated on the two modes"
    assert 0.35 < left < 0.65, "one mode is missing (mode collapse)"
