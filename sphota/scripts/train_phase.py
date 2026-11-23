"""The training phase of the full pipeline as ONE job graph over all GPUs, so no GPU waits while something is ready to run.

    python scripts/train_phase.py --exp experiments --stages "ae step9 codec codec_ablation diffusion train_ablations" --gpus 2

What depends on what (everything else is independent and may run at the same time on the other GPU):
    codec runs        lambda 0.0130 base (150k its, alone: no other run can share it) -> the five fine-tunes (each needs only the base)
    ablation runs     K=1 base -> K=1 fine-tune;  ReLU base -> ReLU fine-tune   (independent of the main chain: they use the GPU that
                      would otherwise idle while the 0.0130 base trains)
    diffusion         needs only the lambda 0.0018 codec: it starts the moment that fine-tune is finished, not after all six
    train_ablations   needs only the lambda 0.0018 codec;    ae, step9: need nothing
The commands are exactly the ones the serial pipeline runs (train_codec for every codec, run_full_pipeline.sh for the stage jobs, so
claims, done-markers, logs and uploads work as before); only the order and the overlap change. A job's output goes to
<exp>/logs/job_<name>.out. Exit status: 0 done, 75 paused by the time budget (state saved), else a crash (the others save and stop).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from truesight import parallel  # noqa: E402
import train_all_lambdas as tal  # noqa: E402  (imports torch first)

HEAVY = ("ae", "step9", "codec", "codec_ablation", "diffusion", "train_ablations")
ABL_LAMBDAS = [0.0130, 0.0035]


def stage_job(name, exp, after, extra_env=None):
    """A pipeline stage run in its own process: ``STAGES=<name> bash scripts/run_full_pipeline.sh`` (TS_INNER stops it from
    starting another job graph). It has its own claim, done-marker, log and upload."""
    env = {"STAGES": name, "TS_INNER": "1"}
    env.update(extra_env or {})
    return parallel.Job(f"stage:{name}", ["bash", str(ROOT / "scripts" / "run_full_pipeline.sh")], list(after),
                        skip_if=Path(exp) / ".done" / name, env=env, log=Path(exp) / "logs" / f"job_{name}.out")


def build_phase_jobs(a, stages):
    """The prioritised job list (list order = priority) for the wanted ``stages``."""
    stages = [s for s in stages if s in HEAVY]
    out, exp = Path(a.codec_dir), Path(a.exp)
    main = tal.build_jobs(out, [(3, "gdn", list(tal.LAMBDAS))], a) if "codec" in stages else []
    abl = tal.build_jobs(out, [(1, "gdn", ABL_LAMBDAS), (3, "relu", ABL_LAMBDAS)], a) if "codec_ablation" in stages else []
    for j in main + abl:                                            # every codec run logs to its own file
        j.log = exp / "logs" / f"job_{j.name}.out"
    by = {j.name: j for j in main + abl}
    lo = by.get("lmbda0.0018")
    lo_dep = [lo.name] if lo else []                                # no dependency when the codec stage is not part of this run
    order = []

    def add(*jobs):
        order.extend(j for j in jobs if j is not None)

    add(by.get("lmbda0.0130"), by.get("K1_lmbda0.0130"), lo)       # the long base runs start first, on both GPUs
    if "diffusion" in stages:
        add(stage_job("diffusion", exp, lo_dep))                    # starts the moment the lambda 0.0018 codec exists
    add(by.get("lmbda0.0483"), by.get("relu_lmbda0.0130"))
    add(*[by.get(n) for n in ("lmbda0.0067", "lmbda0.0035", "lmbda0.0250")])
    add(by.get("K1_lmbda0.0035"), by.get("relu_lmbda0.0035"))
    for s, dep in (("train_ablations", lo_dep), ("ae", []), ("step9", [])):
        if s in stages:
            add(stage_job(s, exp, dep))
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=str(ROOT / "experiments"))
    ap.add_argument("--codec-dir", default=None, help="default <exp>/step7_codec")
    ap.add_argument("--stages", required=True, help="the wanted stages, space separated (only the heavy training ones are used)")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs to use (0 = every visible GPU)")
    ap.add_argument("--base-iters", type=int, default=150_000)
    ap.add_argument("--finetune-iters", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--M", type=int, default=192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the job order and exit")
    a = ap.parse_args()
    a.codec_dir = a.codec_dir or str(Path(a.exp) / "step7_codec")
    jobs = build_phase_jobs(a, a.stages.split())
    ids = parallel.visible_gpus()
    n = a.gpus if a.gpus > 0 else len(ids)
    if a.dry_run:
        for j in jobs:
            print(f"{j.name:24s} after={j.after}")
        return
    if not jobs:
        print("[phase] nothing to train")
        return
    sys.exit(parallel.run_jobs(jobs, gpus=ids[:n]))


if __name__ == "__main__":
    main()
