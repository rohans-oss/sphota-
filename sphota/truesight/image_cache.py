"""Decode every training image once into one raw uint8 file, so a training step reads a crop instead of decoding a 2K PNG.

Why: ``CropDataset`` used to open and fully decode a ~2040 x 1360 PNG (tens of milliseconds, on the CPU) to keep one 128 x 128 or
256 x 256 crop. A trainer needs 8 to 32 such crops per step, the Kaggle machine has 4 CPU cores, so the GPU often waits for the
loader. With the cache a sample is a few contiguous reads.

Nothing about the data changes: the cached pixels are exactly ``Image.open(path).convert("RGB")`` as uint8, and ``CropDataset``
takes the same uniformly random crop and flips with the same probability from them (see data.py), so the sampling distribution and
the tensors are the same as without the cache. Images that are not in the cache (the disk budget ran out, a file changed) are
simply decoded from the PNG as before, so a partial cache is still correct.

Tiled layout (version 2). The cache (about 41 GB for the 4,909 training images) is larger than the RAM of a Kaggle machine (about
29 GB), so part of every random read goes to the disk. A plain (H, W, 3) array makes a 128-pixel crop 128 separate 384-byte reads, one
4 KB page each, scattered 6 KB apart: about 0.5 MB of page-sized random I/O per sample. Each image is therefore stored as TILE x TILE
(128 x 128) tiles, every tile contiguous (49 KB): a crop touches at most 4 tiles (a 256 crop at most 9), each one a sequential
read. The file is memory-mapped with MADV_RANDOM (no useless read-ahead) and opened lazily in every worker process.

Layout of the cache folder:
    data.u8      all images back to back; each image is its tiles in row-major tile order, each tile (th, tw, 3) uint8 in C order
    index.json   {"version": 2, "tile": 128, "bytes": N, "files": {key: {"off": offset, "h": H, "w": W, "size": source file size}}}
The key is the path as written in the file list (relative to the project root). ``index.json`` is written atomically after the
data it points at, so an interrupted build leaves a valid cache that the next build extends.

    python scripts/build_cache.py --list data/processed/train.txt --out data/cache --max-gb auto
Disk: about 8 MB per 2K image (about 41 GB for the 4,909 training images). ``--max-gb auto`` uses the free space minus a reserve.
"""
import json
import mmap
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = 2
TILE = 128
DATA, INDEX = "data.u8", "index.json"
RESERVE_GB = 5.0                      # kept free on the disk for checkpoints, logs and temporary files


def cache_key(path, root):
    """The path as it appears in a file list: relative to ``root`` when it lies below it, else absolute (posix separators)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def load_index(cache_dir):
    try:
        d = json.loads((Path(cache_dir) / INDEX).read_text())
    except (OSError, ValueError):
        return None
    return d if d.get("version") == VERSION and d.get("tile") == TILE else None


def tile_bytes(a, tile=TILE):
    """The bytes of an (H, W, 3) uint8 image in tiled order (see the module docstring)."""
    h, w = a.shape[:2]
    return b"".join(a[y:y + tile, x:x + tile].tobytes() for y in range(0, h, tile) for x in range(0, w, tile))


def tile_offset(entry, ty, tx, tile=TILE):
    """Byte offset (in the data file) of tile (ty, tx) of an image entry, and its (height, width)."""
    h, w = entry["h"], entry["w"]
    th_row = min(tile, h - ty * tile)
    tw = min(tile, w - tx * tile)
    return entry["off"] + ty * tile * w * 3 + tx * tile * th_row * 3, th_row, tw


class ImageCache:
    """Read access to a cache folder. Opens the memory map lazily (so it is cheap to pickle into DataLoader workers)."""

    def __init__(self, cache_dir, root):
        self.dir, self.root = Path(cache_dir), Path(root)
        self.index = load_index(self.dir)
        self._f = self._m = self._buf = None

    @property
    def usable(self):
        return bool(self.index and self.index["files"]) and (self.dir / DATA).exists()             and (self.dir / DATA).stat().st_size >= self.index["bytes"]

    def __len__(self):
        return len(self.index["files"]) if self.index else 0

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_f"] = d["_m"] = d["_buf"] = None
        return d

    def _open(self):
        if self._buf is None:
            self._f = open(self.dir / DATA, "rb")
            self._m = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
            if hasattr(mmap, "MADV_RANDOM"):                      # Linux: no read-ahead for random crop reads
                try:
                    self._m.madvise(mmap.MADV_RANDOM)
                except OSError:
                    pass
            self._buf = np.frombuffer(self._m, dtype=np.uint8)
        return self._buf

    def entry(self, path):
        """The index entry of ``path`` if it is cached and the source file is unchanged, else None."""
        if not self.index:
            return None
        e = self.index["files"].get(cache_key(path, self.root))
        if e is None:
            return None
        try:
            if os.stat(path).st_size != e["size"]:
                return None
        except OSError:
            return None
        return e

    def shape(self, path):
        e = self.entry(path)
        return None if e is None else (e["h"], e["w"])

    def read_region(self, path, top, left, hh, ww):
        """(hh, ww, 3) uint8 copy of a region of the cached image, or None if it is not cached."""
        e = self.entry(path)
        if e is None:
            return None
        h, w = e["h"], e["w"]
        if top < 0 or left < 0 or top + hh > h or left + ww > w:
            raise ValueError(f"region {(top, left, hh, ww)} lies outside the {h}x{w} image")
        buf = self._open()
        out = np.empty((hh, ww, 3), dtype=np.uint8)
        for ty in range(top // TILE, (top + hh - 1) // TILE + 1):
            for tx in range(left // TILE, (left + ww - 1) // TILE + 1):
                off, th, tw = tile_offset(e, ty, tx)
                tile = buf[off: off + th * tw * 3].reshape(th, tw, 3)
                y0, x0 = max(top, ty * TILE), max(left, tx * TILE)
                y1, x1 = min(top + hh, ty * TILE + th), min(left + ww, tx * TILE + tw)
                out[y0 - top: y1 - top, x0 - left: x1 - left] = tile[y0 - ty * TILE: y1 - ty * TILE, x0 - tx * TILE: x1 - tx * TILE]
        return out

    def read_crop(self, path, top, left, size):
        return self.read_region(path, top, left, size, size)

    def get(self, path):
        """The whole (H, W, 3) uint8 image, or None if it is not cached / the source changed."""
        e = self.entry(path)
        return None if e is None else self.read_region(path, 0, 0, e["h"], e["w"])


def _decode(args):
    i, path = args
    a = np.asarray(Image.open(path).convert("RGB"))
    return i, np.ascontiguousarray(a)


def free_bytes(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(p).free


def build_cache(paths, cache_dir, root, max_bytes=None, workers=4, reserve_gb=RESERVE_GB, log=print, flush_every=200):
    """Cache the images in ``paths`` (absolute paths, list order = priority) until ``max_bytes`` is used (None = the free disk
    space minus ``reserve_gb``). Existing valid entries are kept, so this can be run again after an interruption.
    Returns {"cached": n, "total": n, "bytes": n, "skipped_for_space": n}."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx = load_index(cache_dir) or {"version": VERSION, "tile": TILE, "bytes": 0, "files": {}}
    data = cache_dir / DATA
    if not data.exists() or data.stat().st_size < idx["bytes"]:      # the data file is gone or shorter than the index claims
        idx = {"version": VERSION, "tile": TILE, "bytes": 0, "files": {}}
    keys = [cache_key(p, root) for p in paths]
    todo = []
    for i, (p, k) in enumerate(zip(paths, keys)):
        e = idx["files"].get(k)
        if e is None or e["size"] != os.stat(p).st_size:
            todo.append((i, p))
    budget = max_bytes if max_bytes is not None else max(0, free_bytes(cache_dir) - int(reserve_gb * 2 ** 30)) + idx["bytes"]
    log(f"[cache] {len(paths) - len(todo)}/{len(paths)} images already cached; {len(todo)} to do; budget {budget / 2 ** 30:.1f} GB")
    skipped = 0
    with open(data, "r+b" if data.exists() else "wb") as f:
        f.seek(idx["bytes"])
        f.truncate()                                                  # drop bytes written after the last index (interrupted build)
        pending = 0
        with Pool(max(1, workers)) as pool:
            for n, (i, a) in enumerate(pool.imap(_decode, todo, chunksize=2)):
                p = paths[i]
                size = a.nbytes
                if idx["bytes"] + size > budget:
                    skipped += 1                                      # no room: this image keeps being decoded from its PNG
                    continue
                f.write(tile_bytes(a))
                idx["files"][keys[i]] = {"off": idx["bytes"], "h": int(a.shape[0]), "w": int(a.shape[1]), "size": os.stat(p).st_size}
                idx["bytes"] += size
                pending += 1
                if pending >= flush_every:
                    f.flush(); os.fsync(f.fileno()); _write_index(cache_dir, idx); pending = 0
                    log(f"[cache] {len(idx['files'])}/{len(paths)} images, {idx['bytes'] / 2 ** 30:.1f} GB")
        f.flush(); os.fsync(f.fileno())
    _write_index(cache_dir, idx)
    out = {"cached": sum(1 for k in keys if k in idx["files"]), "total": len(paths), "bytes": idx["bytes"], "skipped_for_space": skipped}
    log(f"[cache] done: {out['cached']}/{out['total']} images cached ({out['bytes'] / 2 ** 30:.1f} GB)"
        + (f"; {skipped} did not fit and are decoded from their PNG during training (same data, just slower)" if skipped else ""))
    return out


def _write_index(cache_dir, idx):
    tmp = Path(cache_dir) / (INDEX + ".tmp")
    tmp.write_text(json.dumps(idx))
    os.replace(tmp, Path(cache_dir) / INDEX)
