"""Automatic, resumable dataset download for TrueSight (Step 1).

Every dataset is described by a :class:`Source`. Running ``download_all`` fetches the
archives into ``data/raw/<dataset>/`` (original archives are kept, as the guide asks),
extracts only image files, and for the big "sample-only" sets (text, faces) extracts just
the requested number of images.

Downloads are resumable (``.part`` files + HTTP Range), verified against the server's
Content-Length, retried with back-off, and run in parallel across datasets.
"""
from __future__ import annotations

import io
import random
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

HF = "https://huggingface.co/datasets"
ETH = "https://data.vision.ee.ethz.ch/cvl"


@dataclass
class Source:
    name: str                      # folder name under data/raw/
    role: str                      # train | val | test_kodak | test_clic | test_text | test_face
    kind: str                      # zip | files | parquet
    urls: list[str] = field(default_factory=list)
    sample: int | None = None      # extract only N images (sampled, seeded) from the archive
    note: str = ""
    approx_gb: float = 0.0


SOURCES: list[Source] = [
    Source("kodak", "test_kodak", "files",
           [f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)],
           note="Kodak PhotoCD, 24 images 768x512 (test only)", approx_gb=0.02),
    Source("div2k_valid", "val", "zip", [f"{ETH.replace('https','http')}/DIV2K/DIV2K_valid_HR.zip"],
           note="DIV2K validation, 100 images (used as our val split)", approx_gb=0.45),
    Source("clic_pro_valid", "test_clic", "zip", [f"{ETH}/clic/professional_valid_2020.zip"],
           note="CLIC 2020 professional valid (stand-in for the test split, which is not public)",
           approx_gb=0.13),
    Source("clic_mobile_valid", "test_clic", "zip", [f"{ETH}/clic/mobile_valid_2020.zip"],
           note="CLIC 2020 mobile valid (stand-in for the test split)", approx_gb=0.24),
    Source("celebahq", "test_face", "parquet",
           [f"{HF}/korexyz/celeba-hq-256x256/resolve/main/data/validation-00000-of-00001.parquet"],
           sample=500, note="CelebA-HQ 256x256 validation shard; 500 faces", approx_gb=0.2),
    Source("textocr", "test_text", "zip", [f"{HF}/redactable-llm/TextOCR/resolve/main/test_images.zip"],
           sample=500, note="TextOCR test images (scene text); 500 sampled", approx_gb=0.97),
    Source("div2k_train", "train", "zip", [f"{ETH.replace('https','http')}/DIV2K/DIV2K_train_HR.zip"],
           note="DIV2K train, 800 images 2K res", approx_gb=3.5),
    Source("clic_pro_train", "train", "zip", [f"{ETH}/clic/professional_train_2020.zip"],
           note="CLIC 2020 professional train", approx_gb=2.0),
    Source("clic_mobile_train", "train", "zip", [f"{ETH}/clic/mobile_train_2020.zip"],
           note="CLIC 2020 mobile train", approx_gb=4.0),
    Source("flickr2k", "train", "zip", [f"{HF}/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip"],
           note="Flickr2K, 2650 images (HF mirror of the EDSR release)", approx_gb=11.6),
]
BY_NAME = {s.name: s for s in SOURCES}

# ----------------------------------------------------------------------------- progress
_progress: dict[str, tuple[int, int]] = {}
_lock = threading.Lock()


def _set_progress(key: str, done: int, total: int) -> None:
    with _lock:
        _progress[key] = (done, total)


def _reporter(stop: threading.Event, every: float = 20.0) -> None:
    last = {}
    t0 = time.time()
    while not stop.wait(every):
        with _lock:
            snap = dict(_progress)
        lines = []
        for k, (d, t) in snap.items():
            if t and d < t:
                rate = (d - last.get(k, d)) / every / 1e6
                lines.append(f"  {k:<28} {d/1e6:8.0f}/{t/1e6:.0f} MB  ({100*d/t:4.1f}%)  {rate:5.2f} MB/s")
            last[k] = d
        if lines:
            print(f"[{time.time()-t0:6.0f}s] downloading:\n" + "\n".join(lines), flush=True)


# ----------------------------------------------------------------------------- download
def download_file(url: str, dest: Path, retries: int = 12, chunk: int = 1 << 20) -> Path:
    """Resumable download. Returns dest. Skips if a complete file is already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    key = dest.name
    if dest.exists() and dest.stat().st_size > 0:
        _set_progress(key, dest.stat().st_size, dest.stat().st_size)
        return dest
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True) as r:
                if r.status_code == 416:  # range past EOF -> already complete
                    part.replace(dest)
                    return dest
                r.raise_for_status()
                if have and r.status_code == 200:  # server ignored Range -> restart
                    have = 0
                    part.unlink(missing_ok=True)
                total = have + int(r.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as f:
                    got = have
                    for blk in r.iter_content(chunk):
                        f.write(blk)
                        got += len(blk)
                        _set_progress(key, got, total)
                if total and part.stat().st_size != total:
                    raise IOError(f"incomplete: {part.stat().st_size}/{total}")
            part.replace(dest)
            _set_progress(key, dest.stat().st_size, dest.stat().st_size)
            return dest
        except Exception as e:  # network hiccups of any kind (resume from the .part file)
            wait = min(60, 2 ** attempt)
            print(f"  ! {key}: {type(e).__name__}: {e!r} (retry {attempt+1}/{retries} in {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


# ----------------------------------------------------------------------------- extraction
def _is_image(name: str) -> bool:
    n = name.replace("\\", "/")
    return Path(n).suffix.lower() in IMG_EXT and "__MACOSX" not in n and not Path(n).name.startswith(".")


def extract_zip(archive: Path, out_dir: Path, sample: int | None = None, seed: int = 0) -> int:
    marker = out_dir / ".extracted"
    if marker.exists():
        return sum(1 for p in out_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if _is_image(n))
        if sample and len(names) > sample:
            names = sorted(random.Random(seed).sample(names, sample))
        for n in names:
            target = out_dir / Path(n.replace("\\", "/")).name   # flatten
            if not target.exists():
                target.write_bytes(z.read(n))
    marker.write_text(f"{len(names)} images\n")
    return len(names)


def extract_parquet(archive: Path, out_dir: Path, sample: int, seed: int = 0) -> int:
    import pyarrow.parquet as pq
    marker = out_dir / ".extracted"
    if marker.exists():
        return sum(1 for p in out_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(archive)
    col = "image" if "image" in table.column_names else table.column_names[0]
    rows = table.column(col).to_pylist()
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(sample, len(rows))))
    from PIL import Image
    for k, i in enumerate(idx):
        im = Image.open(io.BytesIO(rows[i]["bytes"]))
        im.convert("RGB").save(out_dir / f"face_{k:04d}.png")
    marker.write_text(f"{len(idx)} images\n")
    return len(idx)


# ----------------------------------------------------------------------------- orchestrator
def fetch_source(src: Source) -> dict:
    d = RAW / src.name
    t0 = time.time()
    if src.kind == "files":
        for u in src.urls:
            download_file(u, d / Path(u).name)
        n = len(list(d.glob("*.png")))
    else:
        arch = download_file(src.urls[0], d / "_archives" / Path(src.urls[0]).name)
        if src.kind == "zip":
            n = extract_zip(arch, d / "images", src.sample)
        else:
            n = extract_parquet(arch, d / "images", src.sample or 500)
    return {"name": src.name, "images": n, "seconds": round(time.time() - t0, 1)}


def download_all(only: list[str] | None = None, skip: list[str] | None = None, workers: int = 4) -> list[dict]:
    chosen = [s for s in SOURCES if (not only or s.name in only) and not (skip and s.name in skip)]
    print("Datasets to fetch:")
    for s in chosen:
        print(f"  - {s.name:<18} ~{s.approx_gb:5.2f} GB  {s.note}")
    RAW.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=_reporter, args=(stop,), daemon=True).start()
    results = []
    try:
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fetch_source, s): s for s in chosen}
            for f in as_completed(futs):
                s = futs[f]
                try:
                    r = f.result()
                    print(f"[done] {r['name']}: {r['images']} images in {r['seconds']}s", flush=True)
                    results.append(r)
                except Exception as e:  # keep going; report at the end
                    print(f"[FAIL] {s.name}: {e}", flush=True)
                    results.append({"name": s.name, "error": str(e)})
    finally:
        stop.set()
    return results
