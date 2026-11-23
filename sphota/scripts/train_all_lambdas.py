"""Step 7 training schedule: all six lambdas + the K = 1 ablation.

    python scripts/train_all_lambdas.py                       # guide: 150k its for 0.0130, then 50k fine-tunes
    python scripts/train_all_lambdas.py --base-iters 2000 --finetune-iters 500 --batch 4 --crop 128 --N 32 --M 48 \
                                        --workers 0           # tiny CPU smoke run
    python scripts/train_all_lambdas.py --from-scratch        # every lambda trained independently (needs more compute)

Order: (1) lambda 0.0130 with K=3 (full run), (2) the other five K=3 lambdas fine-tuned from it, (3) K=1 ablation (A1) at
two lambdas (default 0.0130 and 0.0035), the K=1 base trained like step (1), the other fine-tuned from it, (4) ReLU ablation
(A2: ReLU instead of GDN, K=3) at the same two lambdas, trained the same way.
Every run resumes from its own last.pt and is skipped when its final.pt already exists. If the session time budget
(TS_TIME_LIMIT_HOURS) runs out, the current run saves its state and the script exits with status 75: run it again later.
GPUs: with more than one visible GPU (Kaggle "T4 x2") the runs that do not need each other train side by side, one per GPU
(``--gpus N``, default: every visible GPU; ``--gpus 1`` = strictly one after the other). The order that must stay serial is kept:
a base model (150k iterations) is trained first, and only then the fine-tunes that start from it. The K=1 and ReLU chains do not
need the main chain, so their base models train at the same time as each other. Each run still writes only its own folder.
Layout under --out: lmbda<L>/ (K=3, GDN), K1_lmbda<L>/ (K=1) and relu_lmbda<L>/ (ReLU), each with log.csv, last.pt, final.pt.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight import runtime  # noqa: E402
import torch  # noqa: E402  (torch first: avoids the Windows sklearn/lpips DLL clash)
from truesight import parallel  # noqa: E402
from truesight.codec.train_codec import train_codec  # noqa: E402
from truesight.codec.vae_codec import LAMBDAS  # noqa: E402

BASE = 0.0130


def run_dir(out, lmbda, K, act="gdn"):
    if act != "gdn":
        return out / f"{act}_lmbda{lmbda:.4f}"
    return out / (f"lmbda{lmbda:.4f}" if K == 3 else f"K{K}_lmbda{lmbda:.4f}")


def finetune_order(lams, base=BASE):
    """Base lambda first (it is trained from scratch), then the lowest lambda (the diffusion decoder needs it), then the highest
    (the protected regions need it), then the rest by distance to the base."""
    rest = sorted((l for l in lams if l != min(lams, key=lambda v: abs(v - base))), key=lambda l: abs(l - base))
    first = min(lams, key=lambda v: abs(v - base))
    out = [first]
    for special in (min(lams), max(lams)):
        if special in rest:
            rest.remove(special)
            out.append(special)
    return out + rest


def build_jobs(out, chains, a, python=sys.executable):
    """One parallel.Job per (K, act, lambda). ``chains`` = [(K, act, lambdas)]. A chain's base model has no dependency; every other
    lambda of the chain waits for that base. Returns the job list (priority order: chain by chain, base first)."""
    jobs = []
    for K, act, lams in chains:
        order = finetune_order(lams)
        base_name = None
        for lmbda in order:
            d = run_dir(out, lmbda, K, act)
            scratch = a.from_scratch or base_name is None
            cmd = [python, "-m", "truesight.codec.train_codec", "--lmbda", str(lmbda), "--out", str(d), "--K", str(K), "--act", act,
                   "--iters", str(a.base_iters if scratch else a.finetune_iters), "--N", str(a.N), "--M", str(a.M),
                   "--batch", str(a.batch), "--crop", str(a.crop), "--workers", str(a.workers), "--ckpt-every", str(a.ckpt_every)]
            if a.train_list:
                cmd += ["--train-list", str(a.train_list)]
            if not scratch:
                cmd += ["--init-from", str(run_dir(out, order[0], K, act) / "final.pt")]
            jobs.append(parallel.Job(d.name, cmd, [] if scratch else [base_name], skip_if=d / "final.pt"))
            if base_name is None:
                base_name = d.name
    return jobs


def default_gpus():
    return len(parallel.visible_gpus()) if torch.cuda.is_available() else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step7_codec"))
    ap.add_argument("--base-iters", type=int, default=150_000)
    ap.add_argument("--finetune-iters", type=int, default=50_000)
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    ap.add_argument("--ablation-lambdas", type=float, nargs="*", default=[0.0130, 0.0035])
    ap.add_argument("--no-ablation", action="store_true", help="skip the K=1 (A1) and ReLU (A2) ablations")
    ap.add_argument("--act-ablation-lambdas", type=float, nargs="*", default=[0.0130, 0.0035], help="lambdas of the ReLU run (A2)")
    ap.add_argument("--only-ablation", action="store_true", help="train only the K=1 (A1) and ReLU (A2) runs, not the six main lambdas")
    ap.add_argument("--from-scratch", action="store_true", help="train every lambda for --base-iters, no fine-tuning")
    for k, d in dict(N=128, M=192, batch=8, crop=256, workers=4, ckpt_every=5000).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=d)
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--gpus", type=int, default=0, help="runs trained at the same time, one per GPU (0 = every visible GPU, 1 = serial)")
    a = ap.parse_args()
    out = Path(a.out)
    kw = dict(N=a.N, M=a.M, batch=a.batch, crop=a.crop, workers=a.workers, ckpt_every=a.ckpt_every, train_list=a.train_list)

    def run(lmbda, K, base_final, act="gdn"):
        d = run_dir(out, lmbda, K, act)
        if (d / "final.pt").exists():
            print(f"skip {d.name}: final.pt exists", flush=True)
            return d / "final.pt"
        scratch = a.from_scratch or base_final is None
        train_codec(lmbda, d, K=K, iters=a.base_iters if scratch else a.finetune_iters,
                    init_from=None if scratch else str(base_final), act=act, **kw)
        return d / "final.pt"

    jobs = [] if a.only_ablation else [(3, "gdn", a.lambdas)]
    if not a.no_ablation:
        jobs += [(1, "gdn", a.ablation_lambdas), (3, "relu", a.act_ablation_lambdas)]
    n_gpu = a.gpus if a.gpus > 0 else default_gpus()
    if n_gpu > 1:                                                    # side by side, one run per GPU; bases still go first
        ids = parallel.visible_gpus()[:n_gpu]
        rc = parallel.run_jobs(build_jobs(out, jobs, a), gpus=ids)
        print("done:", sorted(p.parent.name for p in out.glob("*/final.pt")))
        if rc:
            sys.exit(rc)                                             # 75 = paused (state saved), anything else = a crash
        return
    for K, act, lams in jobs:
        base = None
        order = sorted(lams, key=lambda l: abs(l - BASE))            # the base lambda (closest to 0.0130) goes first
        for lmbda in order:
            final = run(lmbda, K, base, act)
            if base is None:
                base = final
    print("done:", sorted(p.parent.name for p in out.glob("*/final.pt")))


if __name__ == "__main__":
    runtime.run_cli(main)
