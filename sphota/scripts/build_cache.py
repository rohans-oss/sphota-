"""Decode the training images once into a raw uint8 cache (see truesight/image_cache.py): training then reads crops instead of
decoding 2K PNGs. Same data and the same random crops as without the cache, only faster. Safe to run again (resumes / extends).

    python scripts/build_cache.py                                   # data/processed/train.txt -> data/cache, as much as fits
    python scripts/build_cache.py --max-gb 30 --workers 4           # at most 30 GB
    python scripts/build_cache.py --list data/processed/train.txt --out /kaggle/temp/cache && export TS_IMAGE_CACHE=/kaggle/temp/cache

``--max-gb auto`` (default) uses the free disk space minus 5 GB. Images that do not fit are decoded from their PNG while training.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.data import read_list  # noqa: E402
from truesight.image_cache import build_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=str(ROOT / "data" / "processed" / "train.txt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--max-gb", default="auto", help="disk budget in GB, or 'auto' (free space minus a 5 GB reserve)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = [f for f in read_list(a.list) if Path(f).exists()]
    if not files:
        sys.exit(f"no images found for {a.list}")
    mb = None if a.max_gb == "auto" else int(float(a.max_gb) * 2 ** 30)
    build_cache(files, a.out, ROOT, max_bytes=mb, workers=a.workers)


if __name__ == "__main__":
    main()
