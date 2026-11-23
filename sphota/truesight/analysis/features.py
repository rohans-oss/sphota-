"""Image complexity features (reused for rate control, hallucination risk and the recommender)."""
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FEATURES = ["grad_mean", "lap_var", "entropy", "edge_density", "colorfulness", "brightness", "jpeg50_bpp"]


def jpeg_bpp(img: Image.Image, quality: int) -> float:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.tell() * 8 / (img.width * img.height)


def image_features(img: Image.Image) -> dict:
    """The seven Step-5 features of a PIL image (computed at native resolution)."""
    img = img.convert("RGB")
    rgb = np.asarray(img)
    g8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = g8.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    hist = np.bincount(g8.ravel(), minlength=256) / g8.size
    ent = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    edges = cv2.Canny(g8, 100, 200)
    f = rgb.astype(np.float32)                    # float: uint8 subtraction would wrap around
    rg, yb = f[..., 0] - f[..., 1], 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return {
        "grad_mean": float(grad.mean()),
        "lap_var": float(cv2.Laplacian(g, cv2.CV_32F).var()),
        "entropy": float(ent),
        "edge_density": float((edges > 0).mean()),
        "colorfulness": float(colorful),
        "brightness": float(g.mean()),
        "jpeg50_bpp": jpeg_bpp(img, 50),
    }


def _row(args):
    path, role, dataset = args
    try:
        img = Image.open(ROOT / path).convert("RGB")
        r = image_features(img)
        r["jpeg10_bpp"] = jpeg_bpp(img, 10)       # regression target for the size predictor
        r.update(path=path, role=role, dataset=dataset, w=img.width, h=img.height, ok=True)
    except Exception as e:
        r = dict(path=path, role=role, dataset=dataset, ok=False, error=str(e))
    return r


def extract_all(out_csv=PROC / "features.csv", workers=3, batch=300) -> pd.DataFrame:
    """Features for every image in the split lists. Resumable: already-done paths are skipped."""
    roles = {}
    for p in sorted(PROC.glob("*.txt")):
        for line in p.read_text().splitlines():      # NOT .split(): CLIC file names contain spaces
            if line.strip():
                roles[line.strip()] = p.stem
    meta = pd.read_csv(PROC / "meta.csv", usecols=["path", "dataset"]).set_index("path")["dataset"].to_dict()
    df = pd.read_csv(out_csv) if Path(out_csv).exists() else pd.DataFrame()
    if len(df):                                       # drop failed rows and rows no longer in any split list
        df = df[df["ok"].astype(bool) & df["path"].isin(roles)].reset_index(drop=True)
    done = set(df["path"]) if len(df) else set()
    todo = [(p, r, meta.get(p, "")) for p, r in roles.items() if p not in done]
    if todo:
        print(f"Extracting features for {len(todo)} images ({len(done)} cached) ...", flush=True)
    for i in range(0, len(todo), batch):
        with ProcessPoolExecutor(workers, max_tasks_per_child=50) as ex:
            rows = list(ex.map(_row, todo[i:i + batch], chunksize=4))
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(out_csv, index=False)
        print(f"  features {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return df
