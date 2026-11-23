"""Run one Kaggle session of the full pipeline (see KAGGLE.md). Put this in the notebook's only cell:

    !python scripts/kaggle_run.py --sync hf:YOUR_USER/sphota-run --data-repo YOUR_USER/sphota-data

It restores the previous run state, runs every unfinished stage until 11.5 hours are used (Kaggle stops at 12), saves everything
and exits. Start the notebook again and it continues exactly where it stopped; after a crash the state at the moment of the crash
is kept, so after you fix the bug it continues from that step, not from zero.

Exit status: 0 all stages finished, 75 paused at the time limit (run again), 76 not enough weekly GPU quota left (not started),
anything else: crashed (fix and run again). Always pass your quota:  --weekly-quota 30   (or  --quota-hours <hours left on the Kaggle page>).
Needs the Hugging Face token as the environment variable HF_TOKEN or as a Kaggle secret named HF_TOKEN (Add-ons > Secrets).
With --chain the next notebook starts by itself when this session ends (see truesight/chain.py and KAGGLE.md, "Automatic restarts").
For a quick local check use --sync dir:/some/folder, which needs no account.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight import kaggle  # noqa: E402


def chain_config(a):
    from truesight import chain
    reset = None if a.quota_reset_dow is None else (a.quota_reset_dow, a.quota_reset_hour)
    return chain.ChainConfig(owner=a.chain_owner or "", gpu_slug=a.chain_gpu_slug, cpu_slug=a.chain_cpu_slug, wait_slug=a.chain_wait_slug,
                             kind=a.chain_kind, count=a.chain_count, max_chain=a.max_chain, repo=a.chain_repo,
                             machine_shape=a.machine_shape, reset=reset, wait_until=a.wait_until, weekly_quota=a.weekly_quota,
                             quota_hours=a.quota_hours, hours_limit=a.hours, dead_hours=a.dead_session_hours)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync", required=True, help="where the run state lives: hf:<user>/<repo> (private model repo) or dir:<path>")
    ap.add_argument("--hours", type=float, default=kaggle.DEFAULT_HOURS, help="time budget of this session (Kaggle limit is 12)")
    ap.add_argument("--data-repo", default=None, help="Hugging Face dataset repo <user>/<name> that keeps data/raw between sessions")
    ap.add_argument("--data-input", default=None, help="folder of an attached Kaggle dataset containing raw/ (and processed/)")
    ap.add_argument("--data-dir", default=None, help="roomy scratch folder to use as data/ (Kaggle: /kaggle/temp/sphota_data)")
    ap.add_argument("--stages", default=None, help='only these stages, for example "diffusion diffusion_eval"')
    ap.add_argument("--smoke", action="store_true", help="tiny settings (same commands, minutes instead of days)")
    ap.add_argument("--sync-every", type=float, default=kaggle.DEFAULT_SYNC_MIN, help="minutes between uploads while running (checkpoints are written more often)")
    ap.add_argument("--quota-hours", type=float, default=None, help="hours of GPU quota the Kaggle page shows as left RIGHT NOW; the session budget becomes this minus 0.5 h")
    ap.add_argument("--weekly-quota", type=float, default=None, help="your weekly GPU quota (for example 30): the driver counts the hours of the last 7 days itself from a ledger in the run state")
    ap.add_argument("--quota-reset-dow", type=int, default=None, help="weekday of the weekly quota reset, Monday = 0 (UTC): the ledger then counts since the last reset, and a waiter notebook can sleep until it")
    ap.add_argument("--quota-reset-hour", type=int, default=0, help="UTC hour of the weekly quota reset")
    ap.add_argument("--claims", action="store_true", help="two notebooks work on the same run repo at the same time: claim every stage first so they never run the same one (give them different --stages)")
    ap.add_argument("--dead-session-hours", type=float, default=None, help="a cancelled or killed session leaves no end time: assume it used this many GPU hours (default: its whole 11.5 h reservation, which over-counts). Example: 0.5")
    ap.add_argument("--no-git-pull", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    ch = ap.add_argument_group("chaining (truesight/chain.py): start the next notebook automatically")
    ch.add_argument("--chain", action="store_true", help="start the next notebook when this session ends (needs KAGGLE_USERNAME + KAGGLE_KEY or KAGGLE_API_TOKEN as notebook secrets, Internet on)")
    ch.add_argument("--chain-kind", choices=["cpu_data", "gpu", "wait"], default="gpu", help="what this notebook is (cpu_data hands over to gpu when the data is ready; wait sleeps until the quota reset)")
    ch.add_argument("--chain-count", type=int, default=0)
    ch.add_argument("--max-chain", type=int, default=40, help="most chained starts, a safeguard against loops")
    ch.add_argument("--chain-owner", default=None, help="your Kaggle username (owner of the notebooks)")
    ch.add_argument("--chain-gpu-slug", default="sphota-train")
    ch.add_argument("--chain-cpu-slug", default="sphota-data")
    ch.add_argument("--chain-wait-slug", default="sphota-wait")
    ch.add_argument("--chain-repo", default="https://github.com/MANOFHATERS/sphota.git", help="repository the next notebook clones")
    ch.add_argument("--machine-shape", default="NvidiaTeslaT4", help="machine_shape of the GPU notebook ('' to omit)")
    ch.add_argument("--wait-until", type=float, default=None, help="(wait notebooks) epoch seconds to sleep until")
    a = ap.parse_args()
    argv = sys.argv[1:]
    kaggle.load_kaggle_secrets()
    if a.sync.startswith("hf:") and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        sys.exit("HF_TOKEN not found: nothing would be saved, so this session does not start (it would waste quota). In the Kaggle editor open "
                 "this notebook > Add-ons > Secrets, create HF_TOKEN (a Hugging Face token with write access) and tick it for this notebook, "
                 "then run again. A notebook that was started by the chain needs the secrets attached once, by hand.")
    if a.chain:
        from truesight import chain as _chain
        if not _chain.have_kaggle_credentials():
            print("WARNING: --chain is on but no Kaggle API credentials were found (secrets KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN, "
                  "attached to this notebook). The session runs, but the next one will NOT start by itself.", flush=True)
    if a.chain_kind == "wait":
        from truesight import chain
        chain.run_wait(chain_config(a), argv, a.sync, Path("/tmp/chain_kernels") if Path("/tmp").exists() else ROOT / "chain_kernels")
        sys.exit(0)
    env = {}
    if a.stages:
        env["STAGES"] = a.stages
    if a.smoke:
        env["SMOKE"] = "1"
    if a.claims:
        env["TS_CLAIMS"] = "1"
    reset = None if a.quota_reset_dow is None else (a.quota_reset_dow, a.quota_reset_hour)
    gpu = False if a.chain_kind == "cpu_data" else None
    status, rc = kaggle.run_session(ROOT, a.sync, a.hours, data_input=a.data_input, data_dir=a.data_dir, data_repo=a.data_repo,
                                    sync_every=a.sync_every, quota_hours=a.quota_hours, weekly_quota=a.weekly_quota, gpu=gpu, reset=reset, dead_hours=a.dead_session_hours,
                                    do_git_pull=not a.no_git_pull, install=not a.no_install, env_extra=env)
    if a.chain:
        from truesight import chain
        ready = True
        if a.chain_kind == "cpu_data":                         # hand over to the GPU only if the data really is there
            ready = kaggle.raw_present(ROOT / "data") and (ROOT / "data" / "processed" / "train.txt").exists()
        chain.after_session(chain_config(a), argv, status, ROOT / "experiments", a.sync,
                            Path("/kaggle/working/chain_kernels") if Path("/kaggle/working").exists() else ROOT / "chain_kernels", ready=ready)
    sys.exit(rc)


if __name__ == "__main__":
    main()
