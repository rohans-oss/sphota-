"""Settings of the web service. Everything is overridable with TS_* environment variables, nothing is hard-coded to a machine.

    TS_EXPERIMENTS   folder with the trained checkpoints (default <repo>/experiments)
    TS_DEVICE        cuda | cpu (default: cuda when available)
    TS_JOBS_DIR      where uploads and decoded images of a session live (default: the system temp folder)
    TS_MAX_PIXELS    larger photos are resized to at most this many pixels before coding; 0 = never resize (default 786432 on a GPU, 98304 on a CPU)
    TS_LIGHT_STEPS   DDIM steps of the "balanced" mode (default 5, as in scripts/build_mode_table.py)
    TS_SEEDS         seeds used for the uncertainty map (default 3, truesight.analysis.hallucination.N_SEEDS)
    TS_CORS          comma separated origins allowed to call the API (default http://localhost:3000,http://127.0.0.1:3000)
    TS_PRELOAD       1 = load the models in the background when the server starts (default 1)
"""
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODES = ("tiny", "small", "balanced", "exact")                 # quality order, same as truesight.analysis.rules.MODES
MODE_LAMBDA = {"tiny": 0, "small": 1, "balanced": 3, "exact": 5}   # index into LAMBDAS, same as recommender.MODE_SPECS
LAMBDAS = (0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483)     # same as truesight.codec.vae_codec.LAMBDAS (checked on load)
ROI_LAMBDA = 5                                                 # the protected regions use the highest-quality codec
NETWORKS = ("2G", "3G", "4G", "WiFi")
DEFAULT_SEED = 7                                               # the diffusion seed written into every file
# used when experiments/step12_guidance/frozen.json is absent (the values of FaithfulGuide / sphota_decode defaults)
DEFAULT_GUIDANCE = dict(w=1.5, steps=25, lr=0.05, t_start=700, alpha=1.0, beta=10.0, inner_steps=3, margin=0.05)


def _env(name, default=None):
    return os.environ.get(f"TS_{name}", default)


def _default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class Settings:
    experiments: Path = field(default_factory=lambda: Path(_env("EXPERIMENTS", ROOT / "experiments")))
    jobs_dir: Path = field(default_factory=lambda: Path(_env("JOBS_DIR", Path(tempfile.gettempdir()) / "sphota_jobs")))
    device: str = field(default_factory=lambda: _env("DEVICE") or _default_device())
    max_pixels: int = field(default_factory=lambda: int(_env("MAX_PIXELS", 768 * 512 if _default_device() == "cuda" else 384 * 256)))
    light_steps: int = field(default_factory=lambda: int(_env("LIGHT_STEPS", 5)))
    cpu_steps: int = field(default_factory=lambda: int(_env("CPU_STEPS", 0)))   # 0 = use the frozen steps; >0 caps them (CPU demo)
    n_seeds: int = field(default_factory=lambda: int(_env("SEEDS", 3)))
    cors: list = field(default_factory=lambda: [o.strip() for o in _env("CORS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()])
    preload: bool = field(default_factory=lambda: _env("PRELOAD", "1") != "0")
    job_ttl_s: int = field(default_factory=lambda: int(_env("JOB_TTL", 6 * 3600)))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", 25)))

    # ---- where the trained things are (the layout of scripts/run_full_pipeline.sh)
    def codec_path(self, lambda_index):
        return self.experiments / "step7_codec" / f"lmbda{LAMBDAS[lambda_index]:.4f}" / "final.pt"

    @property
    def diffusion_path(self):
        """final.pt when the diffusion run has finished, otherwise the latest checkpoint (last.pt holds the EMA weights too)."""
        d = self.experiments / "step10_diffusion"
        return d / "final.pt" if (d / "final.pt").exists() else d / "last.pt"

    @property
    def frozen_path(self):
        return self.experiments / "step12_guidance" / "frozen.json"

    @property
    def serving_dir(self):
        return self.experiments / "serving"

    def guidance(self):
        """The frozen guidance settings (w, steps, lr, t_start, alpha, beta, inner_steps, margin)."""
        g = dict(DEFAULT_GUIDANCE)
        if self.frozen_path.exists():
            g.update({k: v for k, v in json.loads(self.frozen_path.read_text()).items() if k in g})
        return g

    def steps_for(self, mode, guidance=None):
        """DDIM steps of a mode: tiny / small use the frozen steps, balanced the light steps, exact decodes with the codec only."""
        if mode == "exact":
            return 0
        n = self.light_steps if mode == "balanced" else int((guidance or self.guidance())["steps"])
        return min(n, self.cpu_steps) if self.cpu_steps > 0 else n

    def required_files(self):
        """Checkpoints without which no photo can be coded: the four mode codecs (lambda 0, 1, 3, 5) and the diffusion model."""
        lams = sorted(set(MODE_LAMBDA.values()) | {ROI_LAMBDA})
        return [self.codec_path(i) for i in lams] + [self.diffusion_path]

    def optional_files(self):
        """Artifacts of the intelligence and honesty layers (written by scripts/export_serving_models.py)."""
        d = self.serving_dir
        return {"recommender": d / "recommender.joblib", "size_model": d / "size_model.joblib",
                "risk_model": d / "risk_full.joblib", "failure_rules": d / "failure_rules.json"}
