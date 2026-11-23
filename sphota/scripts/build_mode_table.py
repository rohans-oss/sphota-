"""Step 14 data: run every mode on every image and record what happened.

    python scripts/build_mode_table.py --list data/processed/train.txt --limit 1500          # estimated sizes, codec decode
    python scripts/build_mode_table.py ... --real-bytes                                      # real .sphota sizes (needs torchac)
    python scripts/build_mode_table.py ... --diffusion experiments/step10_diffusion/final.pt --ocr --faces
                                                                                             # generative modes + TFR / FIR

For each image (centre crop of at most --crop x --crop pixels, so a 1,500-image table stays affordable) and for each of the six
lambda checkpoints in --codecs-dir (lmbda0.0018 ... lmbda0.0483; modes 0..5) it records the real or estimated file size and the
LPIPS / PSNR of the codec decode, together with the Step 5 image features. Writes into --out:
  mode_table.csv   one row per (image, mode 0..5): image, mode, w, h, the 7 features, bpp, bytes, lpips, psnr      -> rate control
  context.csv      one row per image: features, text_frac, face_frac, text_h, n_faces (detectors on the crop)       -> recommender
  outcomes.csv     one row per (image, mode name): bytes, lpips, text_ok, face_ok, decode_s, decoder_used            -> oracle / rules
The four recommender modes use lambda indices 0, 1, 3, 5. tiny / small decode with diffusion (--steps), balanced with light
diffusion (--light-steps), exact with the codec only. Without --diffusion every mode is evaluated with the codec decode and
``decoder_used`` says so; text_ok / face_ok are NaN unless --ocr / --faces are given (NaN = nothing to check, counts as ok).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from truesight.metrics import lpips_dist, psnr  # noqa: E402  (imports lpips before sklearn)
from truesight.analysis.features import image_features, jpeg_bpp  # noqa: E402
from truesight.analysis.recommender import MODE_SPECS  # noqa: E402
from truesight.analysis.rules import MODES  # noqa: E402
from truesight.codec import bitstream as bs  # noqa: E402
from truesight.codec.train_codec import load_codec  # noqa: E402
from truesight.codec.vae_codec import LAMBDAS, bpp_terms  # noqa: E402
from truesight.data import read_list  # noqa: E402

PROC = ROOT / "data" / "processed"
HEADER_BYTES = bs.header_bytes(0)


def to_t(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


def center_crop(pil, size):
    w, h = pil.size
    s = min(size, w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return pil.crop((x0, y0, x0 + s, y0 + s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=str(PROC / "train.txt"))
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--codecs-dir", default=str(ROOT / "experiments" / "step7_codec"))
    ap.add_argument("--real-bytes", action="store_true", help="write real .sphota files (torchac) instead of estimating the size")
    ap.add_argument("--diffusion", default=None, help="diffusion final.pt for the generative modes")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--light-steps", type=int, default=5)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--faces", action="store_true", help="face identity check (VGGFace2 embedder; downloads weights on first use)")
    ap.add_argument("--no-detect", action="store_true", help="skip the text / face detectors (text_frac = face_frac = 0)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step14_intelligence"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    codecs = []
    for lam in LAMBDAS:
        p = Path(a.codecs_dir) / f"lmbda{lam:.4f}" / "final.pt"
        if not p.exists():
            sys.exit(f"missing checkpoint {p}: train all six lambdas with scripts/train_all_lambdas.py first")
        net, _ = load_codec(p, dev)
        if a.real_bytes:
            net.prepare_for_coding()
        codecs.append(net)
    unet = sched = None
    if a.diffusion:
        from truesight.diffusion.decode import diffusion_decode
        from truesight.diffusion.schedule import Schedule
        from truesight.diffusion.train_diffusion import load_unet
        unet, ucfg = load_unet(a.diffusion, dev)
        sched = Schedule(ucfg["T"], ucfg["kind"], device=dev)
    td = fd = ocr = emb = None
    if not a.no_detect:
        from truesight.protect.detect import FaceDetector, TextDetector
        td, fd = TextDetector(), FaceDetector()
    if a.ocr:
        from truesight.protect.faithfulness import EasyOCRAdapter
        ocr = EasyOCRAdapter()
    if a.faces:
        from truesight.protect.faithfulness import FaceEmbedder
        emb = FaceEmbedder()

    mode_rows, ctx_rows, out_rows = [], [], []
    for n, path in enumerate(read_list(a.list)[: a.limit]):
        pil = center_crop(Image.open(path).convert("RGB"), a.crop)
        w, h = pil.size
        x = to_t(pil).to(dev)
        feats = image_features(pil)
        text = list(td.detect(pil)) if td else []
        faces = list(fd.detect(pil)) if fd else []
        th = [b[3] - b[1] for b in text]
        ctx_rows.append(dict(image=n, name=Path(path).name, w=w, h=h, **feats,
                             text_frac=min(1.0, sum((b[2] - b[0]) * (b[3] - b[1]) for b in text) / (w * h)),
                             face_frac=min(1.0, sum((b[2] - b[0]) * (b[3] - b[1]) for b in faces) / (w * h)),
                             text_h=float(np.median(th)) if th else float("nan"), n_faces=len(faces)))
        decoded, sizes = {}, {}
        for m, net in enumerate(codecs):
            xp = bs.pad_to_multiple(x, net.DOWN)
            with torch.no_grad():
                if a.real_bytes:
                    blob, _ = bs.encode_image(net, x, mode=m)
                    nbytes = len(blob)
                else:
                    nbytes = HEADER_BYTES + sum(bpp_terms(net(xp)["likelihoods"], 1).values()).item() / 8.0
                coarse = net.g_s(torch.round(net.g_a(xp))).clamp(0, 1)[..., :h, :w]
            decoded[m], sizes[m] = coarse, nbytes
            mode_rows.append(dict(image=n, mode=m, w=w, h=h, **feats, bpp=8 * nbytes / (w * h), bytes=nbytes,
                                  lpips=lpips_dist(x.cpu(), coarse.cpu()), psnr=psnr(x.cpu(), coarse.cpu())))
        for name in MODES:
            spec = MODE_SPECS[name]
            lam_i = spec["lambda_index"]
            t0 = time.perf_counter()
            dec, used = decoded[lam_i], "codec"
            if unet is not None and spec["decoder"] != "codec":
                steps = a.steps if spec["decoder"] == "diffusion" else a.light_steps
                dec = diffusion_decode(unet, sched, decoded[lam_i].to(dev), steps=steps, w=a.w, seed=0).cpu()
                used = spec["decoder"]
            dt = time.perf_counter() - t0
            text_ok = face_ok = float("nan")
            if ocr is not None:
                from truesight.protect.faithfulness import text_faithfulness
                r = text_faithfulness(ocr, pil, to_pil(dec))
                text_ok = float("nan") if r is None else float(r["passed"])
            if emb is not None and faces:
                from truesight.protect.faithfulness import face_identity
                fr = face_identity(fd, emb, pil, to_pil(dec))
                face_ok = float("nan") if not fr else float(all(f["passed"] for f in fr))
            out_rows.append(dict(image=n, mode=name, bytes=sizes[lam_i], lpips=lpips_dist(x.cpu(), dec.cpu()), text_ok=text_ok,
                                 face_ok=face_ok, decode_s=dt if used != "codec" else float("nan"), decoder_used=used))
        if (n + 1) % 25 == 0:
            print(f"{n + 1} images", flush=True)
            pd.DataFrame(mode_rows).to_csv(out / "mode_table.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(out / "mode_table.csv", index=False)
    pd.DataFrame(ctx_rows).to_csv(out / "context.csv", index=False)
    pd.DataFrame(out_rows).to_csv(out / "outcomes.csv", index=False)
    print(f"done: {len(ctx_rows)} images, {len(mode_rows)} mode rows -> {out}")


if __name__ == "__main__":
    main()
