"""Copy a Hugging Face repo (the run state, or the dataset) from one account to another, so training resumes instead of restarting.

    HF_TOKEN_OLD = read token of the old account      HF_TOKEN_NEW = write token of the new account     (environment variables only)

    python scripts/copy_hf_run.py --old OLD/sphota-run  --new NEW/sphota-run  --keep-data-markers          # results: checkpoints, logs
    python scripts/copy_hf_run.py --old OLD/sphota-data --new NEW/sphota-data --type dataset               # the 45 GB of images

Best run on a Kaggle CPU notebook (datacenter speed, no GPU quota): put the tokens in Kaggle secrets, point --workdir at /kaggle/temp
(the 20 GB limit of /kaggle/working does not apply there). A re-run continues where an interrupted copy stopped.

Left out of the run-state copy, on purpose:
  * quota/        the weekly GPU ledger belongs to the OLD Kaggle account; a new Kaggle account must count from zero.
  * .running/     leftover "session is running" markers of the old account.
  * .done/data*   only when --keep-data-markers is NOT given (use that flag when the dataset is copied too, so the data stage is skipped).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

SKIP_RUN = ["quota/*", ".running/*", ".cache/*"]
DATA_MARKERS = [".done/data", ".done/data_uploaded"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="OLD_USER/repo")
    ap.add_argument("--new", required=True, help="NEW_USER/repo")
    ap.add_argument("--type", choices=["model", "dataset"], default="model")
    ap.add_argument("--keep-data-markers", action="store_true", help="keep .done/data and .done/data_uploaded (when the dataset is copied too)")
    ap.add_argument("--workdir", default=None, help="scratch folder for the download (Kaggle: /kaggle/temp/copy); deleted afterwards")
    a = ap.parse_args()
    old_t, new_t = os.environ.get("HF_TOKEN_OLD"), os.environ.get("HF_TOKEN_NEW")
    if not old_t or not new_t:
        sys.exit("set HF_TOKEN_OLD and HF_TOKEN_NEW in the environment first (see the top of this file)")
    from huggingface_hub import HfApi, snapshot_download
    skip = [] if a.type == "dataset" else SKIP_RUN + ([] if a.keep_data_markers else DATA_MARKERS)
    api = HfApi(token=new_t)
    if a.workdir:
        Path(a.workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.workdir) as tmp:
        print(f"downloading {a.old} ({a.type}) ...", flush=True)
        snapshot_download(a.old, repo_type=a.type, token=old_t, local_dir=tmp, ignore_patterns=skip or None)
        api.create_repo(a.new, repo_type=a.type, private=True, exist_ok=True)
        print(f"uploading to {a.new} ...", flush=True)
        if a.type == "dataset":                                    # thousands of files: resumable, batched commits
            api.upload_large_folder(repo_id=a.new, repo_type="dataset", folder_path=tmp, ignore_patterns=[".cache/*"])
        else:
            api.upload_folder(repo_id=a.new, repo_type="model", folder_path=tmp, ignore_patterns=skip or None,
                              commit_message="Copy of the run state from the previous account")
    print(f"done: {a.new}")


if __name__ == "__main__":
    main()
