"""Move the run state (checkpoints, logs, results) from one Hugging Face account to another, so training resumes instead of restarting.

    set HF_TOKEN_OLD=<read token of the old account>        (PowerShell: $env:HF_TOKEN_OLD = "...")
    set HF_TOKEN_NEW=<write token of the new account>
    python scripts/copy_hf_run.py --old OLD_USER/sphota-run --new NEW_USER/sphota-run

Tokens are read from the environment only (never typed on the command line or stored). The new private repo is created if needed.

What is NOT copied, on purpose:
  * quota/            the weekly GPU ledger belongs to the OLD Kaggle account; with a new Kaggle account the driver must count from zero.
  * .done/data*       the data stage markers: the NEW data repo is empty, so the data stage must run again (the CPU notebook, no GPU quota).
  * .running/         leftover "session is running" markers of the old account.
Everything else (.done markers of finished training stages, step7_codec, step10_diffusion, logs, baselines) is copied as it is.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

SKIP = ["quota/*", ".done/data", ".done/data_uploaded", ".running/*", ".cache/*"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="OLD_USER/sphota-run")
    ap.add_argument("--new", required=True, help="NEW_USER/sphota-run")
    ap.add_argument("--workdir", default=None, help="scratch folder for the download (default: a temp folder, deleted afterwards)")
    a = ap.parse_args()
    old_t, new_t = os.environ.get("HF_TOKEN_OLD"), os.environ.get("HF_TOKEN_NEW")
    if not old_t or not new_t:
        sys.exit("set HF_TOKEN_OLD and HF_TOKEN_NEW in the environment first (see the top of this file)")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=new_t)
    with tempfile.TemporaryDirectory(dir=a.workdir) as tmp:
        print(f"downloading {a.old} ...", flush=True)
        snapshot_download(a.old, repo_type="model", token=old_t, local_dir=tmp, ignore_patterns=SKIP)
        shutil_clean(Path(tmp))
        api.create_repo(a.new, repo_type="model", private=True, exist_ok=True)
        print(f"uploading to {a.new} ...", flush=True)
        api.upload_folder(repo_id=a.new, repo_type="model", folder_path=tmp, ignore_patterns=SKIP,
                          commit_message="Copy of the run state from the previous account")
    print("done. Use --sync hf:%s on the new Hugging Face account." % a.new)


def shutil_clean(root):
    import shutil
    shutil.rmtree(root / ".cache", ignore_errors=True)


if __name__ == "__main__":
    main()
