"""Faithfulness metrics of Section 8.3: Text Faithfulness Rate (TFR) and Face Identity Rate (FIR).

TFR  Run OCR on the original; keep the text regions it finds with confidence >= 0.5. Read the SAME region in the decode.
     An image passes if the character error rate between the two readings is <= 0.05 for ALL such regions.
     TFR = % of images that pass (images without confident text are not eligible and are not counted).
FIR  For every face found in the original (MTCNN, prob >= 0.9, >= 24 px) embed the face crop of the original and of the
     decode (same box). A face passes if the cosine similarity of the embeddings is >= 0.7. FIR = % of faces passing.

The OCR engine and the face embedder are passed in as objects, so the metric logic can be tested with stand-ins and the
real models (EasyOCR recognition model, InceptionResnetV1 / VGGFace2) are only loaded when an adapter is used:
    ocr.read(rgb_array)            -> [(box, text, confidence)]       box = (x0, y0, x1, y1)
    ocr.recognize(rgb_array, box)  -> text read inside ``box``
    embedder.embed(pil_image)      -> 1-D numpy vector
The adapters below download model weights on first use (EasyOCR english_g2.pth ~14 MB, VGGFace2 ~107 MB).
"""
import numpy as np
from PIL import Image

from ..metrics import cer, cosine
from .detect import ImageMemo, _as_pil, _as_rgb_array

MIN_OCR_CONF = 0.5
MAX_CER = 0.05
MIN_COSINE = 0.7
BOX_MARGIN = 0.10          # crop margin around a face box (fraction of its size); identical for original and decode


# --------------------------------------------------------------------------------------------- adapters (real models)
class EasyOCRAdapter:
    """EasyOCR reader / recogniser. Needs the recognition weights (downloaded on first use)."""

    def __init__(self, langs=("en",), gpu=False, reader=None):
        self.langs, self.gpu, self._reader = list(langs), gpu, reader
        self._memo = ImageMemo()                              # the original image is read once, not once per method

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
        return self._reader

    def read(self, img):
        a = _as_rgb_array(img)

        def run():
            out = []
            for poly, text, conf in self.reader.readtext(a):
                pts = np.asarray(poly, dtype=float)
                out.append(((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()), text, float(conf)))
            return out
        return self._memo.get(ImageMemo.key(a), run)

    def recognize(self, img, box):
        a = _as_rgb_array(img)
        h, w = a.shape[:2]
        x0, y0, x1, y1 = box
        region = [max(0, int(x0)), min(w, int(np.ceil(x1))), max(0, int(y0)), min(h, int(np.ceil(y1)))]
        res = self.reader.recognize(a, horizontal_list=[region], free_list=[])
        return " ".join(t for _, t, _ in res)


class FaceEmbedder:
    """InceptionResnetV1 (VGGFace2) face embeddings from facenet-pytorch. Weights download on first use."""

    def __init__(self, device="cpu", model=None):
        self.device, self._model = device, model
        self._memo = ImageMemo()                              # the original's face crops are embedded once, not once per method

    @property
    def model(self):
        if self._model is None:
            from facenet_pytorch import InceptionResnetV1
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        return self._model

    def embed(self, pil):
        pil = pil.convert("RGB")
        return self._memo.get(ImageMemo.key(np.asarray(pil)), lambda: self._embed(pil))

    def _embed(self, pil):
        import torch
        x = torch.from_numpy(np.asarray(pil.convert("RGB").resize((160, 160), Image.BILINEAR), dtype=np.float32))
        x = ((x - 127.5) / 128.0).permute(2, 0, 1)[None].to(self.device)           # facenet "fixed standardisation"
        with torch.no_grad():
            return self.model(x)[0].cpu().numpy()


# --------------------------------------------------------------------------------------------- Text Faithfulness Rate
def text_faithfulness(ocr, original, decoded, min_conf=MIN_OCR_CONF, max_cer=MAX_CER):
    """TFR verdict for one image pair. Returns None if the original has no confident text (not eligible), else
    {"passed": bool, "n_regions": int, "n_ok": int, "cers": [...]}."""
    regions = [(b, t, c) for b, t, c in ocr.read(original) if c >= min_conf and t.strip()]
    if not regions:
        return None
    read = [ocr.recognize(decoded, box) for box, _, _ in regions]
    cers = [float(cer(text, r)) for (_, text, _), r in zip(regions, read)]
    ok = [c <= max_cer for c in cers]
    return {"passed": all(ok), "n_regions": len(regions), "n_ok": int(sum(ok)), "cers": cers,
            "regions": [{"box": tuple(float(v) for v in b), "original": t, "decoded": r, "cer": c}
                        for (b, t, _), r, c in zip(regions, read, cers)]}


def text_faithfulness_rate(ocr, pairs, **kw):
    """TFR over (original, decoded) image pairs: {"tfr": % passing among eligible images, counts, per-image results}."""
    res = [text_faithfulness(ocr, o, d, **kw) for o, d in pairs]
    eligible = [r for r in res if r is not None]
    n_reg = sum(r["n_regions"] for r in eligible)
    return {"tfr": 100.0 * sum(r["passed"] for r in eligible) / len(eligible) if eligible else float("nan"),
            "n_images": len(res), "n_eligible": len(eligible), "n_pass": sum(r["passed"] for r in eligible),
            "region_rate": 100.0 * sum(r["n_ok"] for r in eligible) / n_reg if n_reg else float("nan"),
            "per_image": res}


# --------------------------------------------------------------------------------------------- Face Identity Rate
def _crop(pil, box, margin=BOX_MARGIN):
    x0, y0, x1, y1 = box
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    return pil.crop((max(0, int(x0 - mw)), max(0, int(y0 - mh)), min(pil.width, int(np.ceil(x1 + mw))),
                     min(pil.height, int(np.ceil(y1 + mh)))))


def face_identity(face_detector, embedder, original, decoded, min_cos=MIN_COSINE):
    """FIR verdicts for one image pair: a list of {"box", "cosine", "passed"} (empty if no face is found)."""
    o, d = _as_pil(original), _as_pil(decoded)
    out = []
    for box in face_detector.detect(o):
        c = cosine(embedder.embed(_crop(o, box)), embedder.embed(_crop(d, box)))
        out.append({"box": tuple(float(v) for v in box), "cosine": float(c), "passed": bool(c >= min_cos)})
    return out


def face_identity_rate(face_detector, embedder, pairs, **kw):
    """FIR over (original, decoded) pairs: {"fir": % of faces passing, n_faces, n_pass, n_images_with_faces, per_image}."""
    res = [face_identity(face_detector, embedder, o, d, **kw) for o, d in pairs]
    faces = [f for r in res for f in r]
    return {"fir": 100.0 * sum(f["passed"] for f in faces) / len(faces) if faces else float("nan"),
            "n_faces": len(faces), "n_pass": sum(f["passed"] for f in faces),
            "n_images_with_faces": sum(1 for r in res if r), "n_images": len(res), "per_image": res}
