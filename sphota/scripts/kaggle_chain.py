"""Start the automatic Kaggle chain from your own computer (or any notebook): writes the notebook folder and pushes it with the kaggle CLI.

    # 1. one CPU notebook downloads the data, then starts the GPU notebook, which trains and restarts itself until everything is done
    python scripts/kaggle_chain.py start --kind cpu_data --owner YOUR_KAGGLE_USER -- \
        --sync hf:YOU/sphota-run --data-repo YOU/sphota-data --data-dir /kaggle/temp/sphota_data --stages data \
        --weekly-quota 30 --quota-reset-dow 0 --chain

    # skip the data step if it is already uploaded: start the GPU notebook directly
    python scripts/kaggle_chain.py start --kind gpu --owner YOUR_KAGGLE_USER -- \
        --sync hf:YOU/sphota-run --data-repo YOU/sphota-data --data-dir /kaggle/temp/sphota_data \
        --weekly-quota 30 --quota-reset-dow 0 --chain

    python scripts/kaggle_chain.py start ... --write-only     # only write the notebook folder (to look at it, or push it yourself)
    python scripts/kaggle_chain.py stop-help                  # how to stop the chain

Everything after ``--`` is passed to scripts/kaggle_run.py (the chain flags --chain-kind / --chain-owner / ... are added for you).
Needs the kaggle CLI (pip install kaggle) with your API credentials (~/.kaggle/kaggle.json or the KAGGLE_USERNAME + KAGGLE_KEY /
KAGGLE_API_TOKEN environment variables). The notebooks themselves read HF_TOKEN, KAGGLE_USERNAME + KAGGLE_KEY (or KAGGLE_API_TOKEN) and, for
a private repository, GITHUB_TOKEN from the Kaggle secrets attached to them: after the first push open each notebook once in the editor,
Add-ons > Secrets, and tick those secrets. See KAGGLE.md ("Automatic restarts").
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight import chain  # noqa: E402

STOP_HELP = """To stop the chain: create an empty file named .stop_chain in your Hugging Face run repo (huggingface.co/<you>/<repo> >
Files and versions > Add file > Create a new file). The next step of the chain sees it and stops. Remove the file to allow chaining again.
To stop right now, stop the running notebook in the Kaggle editor."""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start", help="write and push a notebook")
    st.add_argument("--kind", choices=["cpu_data", "gpu", "wait"], required=True)
    st.add_argument("--owner", required=True, help="your Kaggle username")
    st.add_argument("--gpu-slug", default="sphota-train")
    st.add_argument("--cpu-slug", default="sphota-data")
    st.add_argument("--wait-slug", default="sphota-wait")
    st.add_argument("--repo", default=chain.DEFAULT_REPO)
    st.add_argument("--machine-shape", default=chain.DEFAULT_MACHINE_SHAPE, help="machine_shape of the GPU notebook ('' to omit)")
    st.add_argument("--workdir", default=str(ROOT / "chain_kernels"))
    st.add_argument("--write-only", action="store_true")
    st.add_argument("driver_args", nargs=argparse.REMAINDER, help="after --: arguments for scripts/kaggle_run.py")
    sub.add_parser("stop-help", help="how to stop the chain")
    a = ap.parse_args()
    if a.cmd == "stop-help":
        print(STOP_HELP)
        return
    rest = [x for x in a.driver_args if x != "--"]
    argv = chain.next_argv(rest, a.kind, 0)
    if "--chain" not in argv:
        argv.append("--chain")
    for flag, val in (("--chain-owner", a.owner), ("--chain-gpu-slug", a.gpu_slug), ("--chain-cpu-slug", a.cpu_slug),
                      ("--chain-wait-slug", a.wait_slug), ("--chain-repo", a.repo), ("--machine-shape", a.machine_shape)):
        argv = chain.strip_flag(argv, flag) + [flag, val]
    slug = {"gpu": a.gpu_slug, "cpu_data": a.cpu_slug, "wait": a.wait_slug}[a.kind]
    folder = chain.write_kernel(Path(a.workdir) / slug, a.owner, slug, argv, gpu=(a.kind == "gpu"), repo=a.repo, machine_shape=a.machine_shape)
    print(f"notebook folder: {folder}")
    if a.write_only:
        return
    ok, out = chain.push_kernel(folder)
    print(out)
    if not ok:
        sys.exit("push failed: check the kaggle CLI credentials (pip install kaggle; ~/.kaggle/kaggle.json) and the output above")
    print(f"started https://www.kaggle.com/code/{a.owner}/{slug}\n" + STOP_HELP)


if __name__ == "__main__":
    main()
