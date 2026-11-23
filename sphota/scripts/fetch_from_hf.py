"""Download the trained results from Hugging Face into experiments/ so the website serves the real models.

    python scripts/fetch_from_hf.py --repo <user>/<repo> [--type model|dataset] [--revision main] [--token hf_...]

The repo must mirror the experiments/ layout (step7_codec/lmbda*/final.pt, step10_diffusion/final.pt, step12_guidance/frozen.json,
step13_honesty/, step14_intelligence/, serving/). Afterwards, if step14_intelligence/ was downloaded, run
scripts/export_serving_models.py, then restart the API. HF_TOKEN in the environment works instead of --token for a private repo.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["step7_codec/*/final.pt", "step10_diffusion/final.pt", "step10_diffusion/last.pt", "step12_guidance/**",
            "step13_honesty/**", "step14_intelligence/*.csv", "step14_intelligence/*.json", "serving/**"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--type", default="model", choices=("model", "dataset"))
    ap.add_argument("--revision", default=None)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--subdir", default="", help="folder inside the repo that holds experiments/ contents, if not the root")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    pat = [f"{a.subdir.strip('/')}/{p}" if a.subdir else p for p in PATTERNS]
    path = snapshot_download(repo_id=a.repo, repo_type=a.type, revision=a.revision, token=a.token,
                             allow_patterns=pat, local_dir=str(ROOT / "experiments") if not a.subdir else str(ROOT / "experiments" / "_hf"))
    print("downloaded to", path)
    if a.subdir:
        print(f"files are under {path}/{a.subdir}: move them up into experiments/ (same folder names).")
    print("next: python scripts/export_serving_models.py   (if serving/ was not in the repo), then restart the API")


if __name__ == "__main__":
    main()
