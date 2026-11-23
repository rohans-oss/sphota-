"""Scan, filter, de-duplicate and split the downloaded datasets; write file lists + report.

Pipeline (all driven by ``prepare_all``):
  1. scan every image under data/raw/<dataset>/images -> metadata + pHash + quick statistics
  2. filter  (train/val only): drop corrupt, <256 px on a side, grayscale
  3. dedup   (pHash, Hamming <= 4): test sets are never touched; val drops anything that
             matches test; train drops anything that matches test or val, and duplicates inside train
  4. split   -> data/processed/{train,val,test_kodak,test_clic,test_text,test_face}.txt
  5. report  -> report/dataset_report.md, report/dataset_stats.json, report/figures/*.png
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data_download import IMG_EXT, RAW, ROOT, SOURCES

PROC = ROOT / "data" / "processed"
REPORT = ROOT / "report"
MIN_SIDE = 256
HAMMING = 4
SPLITS = ["train", "val", "test_kodak", "test_clic", "test_text", "test_face"]
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ----------------------------------------------------------------------------- scanning
def _analyse(path: str) -> dict:
    """Open one image and return metadata, pHash and cheap quality statistics."""
    import imagehash
    from PIL import Image
    p = Path(path)
    out = {"path": path, "bytes": p.stat().st_size, "ok": False}
    try:
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
            if im.format == "JPEG":
                im.draft("RGB", (512, 512))
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            a = np.asarray(im, dtype=np.float32)
        gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
        rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
        chroma = float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())
        out.update(
            ok=True, w=w, h=h, mode=mode, mp=w * h / 1e6, bpp_orig=8 * out["bytes"] / (w * h),
            aspect=w / h, phash=int(str(imagehash.phash(im)), 16),
            lum_mean=float(gray.mean()), lum_std=float(gray.std()), sharp=float(lap.var()),
            colorful=float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
            is_gray=bool(mode in ("L", "LA", "1", "I;16") or chroma < 2.0),
        )
    except Exception as e:  # corrupt / unreadable
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def scan(force: bool = False, workers: int = 3, batch: int = 300) -> pd.DataFrame:
    """Analyse every image. Few workers (big PNG/JPEG decodes are memory hungry) and the cache
    is checkpointed after every batch, so an interrupted scan resumes instead of restarting."""
    PROC.mkdir(parents=True, exist_ok=True)
    cache = PROC / "meta.csv"
    df = pd.read_csv(cache, dtype={"phash": "uint64"}) if cache.exists() and not force else pd.DataFrame()
    known = set(df["path"]) if len(df) else set()
    todo = []
    for s in SOURCES:
        folder = RAW / s.name if s.kind == "files" else RAW / s.name / "images"
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rel = p.relative_to(ROOT).as_posix()
                if rel not in known:
                    todo.append((s, str(p), rel))
    if todo:
        print(f"Scanning {len(todo)} new images with {workers} workers ...", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        with ProcessPoolExecutor(workers, max_tasks_per_child=40) as ex:
            rows = []
            for (s, _, rel), r in zip(chunk, ex.map(_analyse, [t[1] for t in chunk], chunksize=4)):
                r["path"], r["dataset"], r["role"] = rel, s.name, s.role
                rows.append(r)
        new = pd.DataFrame(rows)
        new["phash"] = new["phash"].fillna(0).astype("uint64") if "phash" in new else 0
        df = pd.concat([df, new], ignore_index=True) if len(df) else new
        df.to_csv(cache, index=False)
        print(f"  scanned {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return df


# ----------------------------------------------------------------------------- dedup helpers
def _hamming_to_set(h: np.uint64, ref: np.ndarray) -> np.ndarray:
    if ref.size == 0:
        return np.zeros(0, dtype=np.int32)
    x = np.bitwise_xor(ref, h).view(np.uint8).reshape(-1, 8)
    return _POP[x].sum(1)


def _near_dup_mask(cands: np.ndarray, ref: np.ndarray, thr: int = HAMMING) -> np.ndarray:
    return np.array([bool((_hamming_to_set(h, ref) <= thr).any()) if ref.size else False for h in cands], dtype=bool)


def _dedup_within(hashes: np.ndarray, thr: int = HAMMING) -> np.ndarray:
    keep = np.ones(len(hashes), bool)
    for i in range(len(hashes)):
        if not keep[i]:
            continue
        d = _hamming_to_set(hashes[i], hashes[i + 1:])
        keep[i + 1:][d <= thr] = False
    return keep


# ----------------------------------------------------------------------------- split
def split(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict]:
    log: dict = {"dropped": {}}
    df = df.copy()
    bad = ~df["ok"].astype(bool)
    log["corrupt"] = int(bad.sum())
    df = df[~bad].copy()
    df["phash"] = df["phash"].astype("uint64")

    test_roles = [r for r in SPLITS if r.startswith("test_")]
    test = df[df["role"].isin(test_roles)]
    test_h = test["phash"].to_numpy(dtype="uint64")

    out: dict[str, pd.DataFrame] = {r: df[df["role"] == r] for r in test_roles}

    def clean(sub: pd.DataFrame, name: str, ref: np.ndarray) -> pd.DataFrame:
        n0 = len(sub)
        small = (sub["w"] < MIN_SIDE) | (sub["h"] < MIN_SIDE)
        gray = sub["is_gray"].astype(bool)
        sub = sub[~small & ~gray]
        log["dropped"][f"{name}_small"] = int(small.sum())
        log["dropped"][f"{name}_gray"] = int((gray & ~small).sum())
        leak = _near_dup_mask(sub["phash"].to_numpy("uint64"), ref)
        log["dropped"][f"{name}_dup_of_held_out"] = int(leak.sum())
        sub = sub[~leak]
        keep = _dedup_within(sub["phash"].to_numpy("uint64"))
        log["dropped"][f"{name}_dup_internal"] = int((~keep).sum())
        sub = sub[keep]
        log[f"{name}_before"], log[f"{name}_after"] = n0, len(sub)
        return sub

    out["val"] = clean(df[df["role"] == "val"], "val", test_h)
    val_h = out["val"]["phash"].to_numpy("uint64")
    out["train"] = clean(df[df["role"] == "train"], "train", np.concatenate([test_h, val_h]))
    lists = {k: out[k]["path"].tolist() for k in SPLITS if k in out}
    return lists, log


def write_lists(lists: dict[str, list[str]]) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    for k, v in lists.items():
        (PROC / f"{k}.txt").write_text("\n".join(v) + ("\n" if v else ""))


def check_no_leakage(lists: dict[str, list[str]]) -> dict:
    train = set(lists.get("train", []))
    res = {}
    for k, v in lists.items():
        if k != "train":
            res[f"train&{k}"] = len(train & set(v))
    return res


# ----------------------------------------------------------------------------- report
def _stats(g: pd.DataFrame) -> dict:
    q = lambda c, p: float(np.percentile(g[c], p))
    return {
        "images": int(len(g)),
        "width_min_med_max": [int(g.w.min()), int(g.w.median()), int(g.w.max())],
        "height_min_med_max": [int(g.h.min()), int(g.h.median()), int(g.h.max())],
        "megapixels_mean": round(float(g.mp.mean()), 2),
        "aspect_median": round(float(g.aspect.median()), 3),
        "portrait_pct": round(float((g.aspect < 1).mean() * 100), 1),
        "file_MB_total": round(float(g.bytes.sum() / 1e6), 1),
        "orig_bpp_median": round(float(g.bpp_orig.median()), 2),
        "luminance_mean": round(float(g.lum_mean.mean()), 1),
        "luminance_std_mean": round(float(g.lum_std.mean()), 1),
        "sharpness_p10_p50_p90": [round(q("sharp", p), 1) for p in (10, 50, 90)],
        "colorfulness_mean": round(float(g.colorful.mean()), 1),
        "grayscale": int(g.is_gray.astype(bool).sum()),
        "below_256px": int(((g.w < MIN_SIDE) | (g.h < MIN_SIDE)).sum()),
        "formats": g.path.str.rsplit(".", n=1).str[-1].str.lower().value_counts().to_dict(),
    }


def make_figures(df: pd.DataFrame, lists: dict[str, list[str]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    figs = REPORT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    d = df[df["ok"].astype(bool)]
    names = [s.name for s in SOURCES if (d.dataset == s.name).any()]
    made = []

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for n in names:
        g = d[d.dataset == n]
        ax[0].scatter(g.w, g.h, s=6, alpha=.5, label=n)
    ax[0].axhline(MIN_SIDE, c="k", ls=":"); ax[0].axvline(MIN_SIDE, c="k", ls=":")
    ax[0].set(xscale="log", yscale="log", xlabel="width px", ylabel="height px", title="Resolution (dotted = 256 px filter)")
    ax[0].legend(fontsize=6)
    ax[1].boxplot([d[d.dataset == n].mp for n in names], tick_labels=names, vert=True)
    ax[1].set(yscale="log", ylabel="megapixels", title="Megapixels per image"); ax[1].tick_params(axis="x", rotation=60, labelsize=7)
    ax[2].boxplot([d[d.dataset == n].bpp_orig.clip(upper=30) for n in names], tick_labels=names)
    ax[2].set(ylabel="bits per pixel of original file (clipped 30)", title="Original bpp"); ax[2].tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout(); fig.savefig(figs / "resolution_overview.png", dpi=110); plt.close(fig)
    made.append("resolution_overview.png")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for c, t, a in (("lum_mean", "Mean luminance", ax[0]), ("sharp", "Sharpness (Laplacian var, 256px thumb, log)", ax[1]),
                    ("colorful", "Colorfulness (Hasler-Susstrunk)", ax[2])):
        a.boxplot([d[d.dataset == n][c] for n in names], tick_labels=names); a.set_title(t)
        a.tick_params(axis="x", rotation=60, labelsize=7)
    ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(figs / "content_stats.png", dpi=110); plt.close(fig)
    made.append("content_stats.png")

    fig, axes = plt.subplots(len(names), 5, figsize=(12, 2.1 * len(names)))
    rng = np.random.default_rng(0)
    for r, n in enumerate(names):
        g = d[d.dataset == n]
        pick = g.iloc[rng.choice(len(g), min(5, len(g)), replace=False)]
        for c in range(5):
            axes[r, c].axis("off")
            if c < len(pick):
                with Image.open(ROOT / pick.iloc[c].path) as im:
                    if im.format == "JPEG":
                        im.draft("RGB", (400, 400))
                    im = im.convert("RGB"); im.thumbnail((300, 300))
                    axes[r, c].imshow(im)
        axes[r, 0].set_title(n, fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(figs / "examples.png", dpi=100); plt.close(fig)
    made.append("examples.png")

    if lists.get("train"):
        from .data import CropDataset
        ds = CropDataset(PROC / "train.txt", crop=256)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, a in enumerate(axes.flat):
            a.imshow(ds[i * max(1, len(ds) // 16)].permute(1, 2, 0).numpy()); a.axis("off")
        fig.tight_layout(); fig.savefig(figs / "train_crops.png", dpi=90); plt.close(fig)
        made.append("train_crops.png")
    return made


def write_report(df: pd.DataFrame, lists: dict, log: dict, leak: dict, figs: list[str]) -> dict:
    REPORT.mkdir(parents=True, exist_ok=True)
    d = df[df["ok"].astype(bool)]
    per_ds = {n: _stats(g) for n, g in d.groupby("dataset")}
    per_split = {}
    for k, v in lists.items():
        g = d[d.path.isin(set(v))]
        per_split[k] = _stats(g) if len(g) else {"images": 0}
    stats = {"per_dataset": per_ds, "per_split": per_split, "filter_log": log, "leakage_check": leak,
             "targets": {"train_ge_3000": len(lists.get("train", [])) >= 3000,
                         "no_test_in_train": all(v == 0 for v in leak.values())}}
    (REPORT / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    L = ["# TrueSight dataset report", "", "Generated by `python scripts/prepare_data.py --stage prepare`.", "",
         "## Split sizes", "", "| split | images | content |", "|---|---|---|"]
    desc = {"train": "DIV2K train + Flickr2K + CLIC train (filtered, de-duplicated)", "val": "DIV2K valid",
            "test_kodak": "Kodak PhotoCD", "test_clic": "CLIC 2020 valid (professional + mobile), stand-in for the non-public test split",
            "test_text": "TextOCR scene-text images", "test_face": "CelebA-HQ faces"}
    for k in SPLITS:
        L.append(f"| {k} | {len(lists.get(k, []))} | {desc[k]} |")
    L += ["", "## Done criteria", "",
          f"- At least 3,000 training images: **{'PASS' if stats['targets']['train_ge_3000'] else 'FAIL'}** ({len(lists.get('train', []))})",
          f"- No test/val image in the training list (exact path + pHash<=4): **{'PASS' if stats['targets']['no_test_in_train'] else 'FAIL'}** `{leak}`",
          "", "## Filtering and de-duplication", "", "```", json.dumps(log, indent=2), "```", "",
          "## Per-dataset statistics", ""]
    cols = ["images", "width_min_med_max", "height_min_med_max", "megapixels_mean", "portrait_pct", "orig_bpp_median",
            "luminance_mean", "colorfulness_mean", "grayscale", "below_256px"]
    L += ["| dataset | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for n, s in per_ds.items():
        L.append(f"| {n} | " + " | ".join(str(s[c]) for c in cols) + " |")
    L += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]
    (REPORT / "dataset_report.md").write_text("\n".join(L) + "\n")
    return stats


def prepare_all(force: bool = False) -> dict:
    df = scan(force)
    if df.empty:
        raise SystemExit("No images found under data/raw/. Run --stage download first.")
    lists, log = split(df)
    write_lists(lists)
    leak = check_no_leakage(lists)
    figs = make_figures(df, lists)
    stats = write_report(df, lists, log, leak, figs)
    print(json.dumps({k: len(v) for k, v in lists.items()}, indent=2))
    print("leakage:", leak, "| targets:", stats["targets"])
    return stats
