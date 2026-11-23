"""Download, clean, de-duplicate and split datasets (Step 1).

    python scripts/prepare_data.py                       # download everything, then prepare
    python scripts/prepare_data.py --stage download      # only download + extract
    python scripts/prepare_data.py --stage prepare       # only filter/dedup/split/report
    python scripts/prepare_data.py --only kodak div2k_valid --skip flickr2k
    python scripts/prepare_data.py --list                # show datasets and sizes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truesight.data_download import SOURCES, download_all  # noqa: E402
from truesight.data_prepare import prepare_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["download", "prepare", "all"], default="all")
    ap.add_argument("--only", nargs="*", help="dataset names to fetch")
    ap.add_argument("--skip", nargs="*", help="dataset names to skip")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--force", action="store_true", help="re-scan every image")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for s in SOURCES:
            print(f"{s.name:<18} {s.role:<11} ~{s.approx_gb:5.2f} GB  {s.note}")
        print(f"total ~{sum(s.approx_gb for s in SOURCES):.1f} GB")
        return
    if a.stage in ("download", "all"):
        res = download_all(a.only, a.skip, a.workers)
        bad = [r for r in res if "error" in r]
        if bad:
            print("Failed:", bad)
            if a.stage == "download":
                sys.exit(1)
    if a.stage in ("prepare", "all"):
        prepare_all(a.force)


if __name__ == "__main__":
    main()
