"""Generates the DEMO images and JSON in public/ (guide F15). Demo values only: no trained model is involved.
The 'plain diffusion' and 'Sphota' images are illustrations of the guide's story (dose 500 mg vs 800 mg).
Run: python tools/make_demo_assets.py"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "public"
for d in ("proof", "demo", "samples", "demo/preview"):
    (PUB / d).mkdir(parents=True, exist_ok=True)

W, H = 1200, 900
F = "C:/Windows/Fonts/"
bold = lambda s: ImageFont.truetype(F + "arialbd.ttf", s)
reg = lambda s: ImageFont.truetype(F + "arial.ttf", s)
rng = np.random.default_rng(3)
boxes = {}


def wood():
    base = np.zeros((H, W, 3), np.float32)
    for y in range(H):
        t = y / H
        base[y, :] = (112 - 24 * t, 76 - 18 * t, 55 - 14 * t)
    grain = rng.normal(0, 1, (H, 1)) * 5 + np.sin(np.arange(H)[:, None] / 7.0) * 4
    img = base + grain[:, :, None] + rng.normal(0, 2.2, (H, W, 1))
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


def render(dose="500 mg", batch="B.No. B2417    Exp 08/2027", record=False):
    im = wood(); d = ImageDraw.Draw(im)
    d.rounded_rectangle([196, 150, 1056, 810], 34, fill=(60, 36, 24))           # soft shadow
    d.rounded_rectangle([160, 110, 1040, 770], 30, fill=(243, 235, 216))         # box face
    d.rounded_rectangle([160, 110, 1040, 200], 30, fill=(201, 86, 30))           # top band
    d.rectangle([160, 150, 1040, 200], fill=(201, 86, 30))
    d.text((200, 128), "RX  ONLY  ·  FILM-COATED", font=bold(34), fill=(245, 231, 198))
    name_xy, dose_xy, batch_xy = (200, 235), (200, 395), (200, 665)
    d.text(name_xy, "PARACETAMOL", font=bold(104), fill=(70, 42, 28))
    d.text((204, 350), "TABLETS IP", font=bold(36), fill=(150, 90, 50))
    d.text(dose_xy, dose, font=bold(190), fill=(201, 86, 30))
    d.text((204, 610), "Each film-coated tablet contains Paracetamol IP 500 mg", font=reg(27), fill=(90, 60, 45))
    d.text(batch_xy, batch, font=bold(36), fill=(70, 42, 28))
    for i in range(5):                                                           # blister foil strip, bottom right
        x, y = 720 + i * 60, 668
        d.ellipse([x, y, x + 46, y + 46], fill=(205, 205, 210), outline=(150, 150, 158), width=3)
    if record:
        bb = lambda xy, t, f: d.textbbox(xy, t, font=f)
        for key, (xy, t, f) in {"name": (name_xy, "PARACETAMOL", bold(104)), "dose": (dose_xy, dose, bold(190)),
                                "batch": (batch_xy, batch, bold(36))}.items():
            x0, y0, x1, y1 = bb(xy, t, f)
            boxes[key] = (x0 - 12, y0 - 10, x1 + 12, y1 + 10)
    return im


def genish(im):
    """Soft generative-looking finish applied identically to the plain and Sphota images."""
    a = im.filter(ImageFilter.GaussianBlur(1.1))
    arr = np.asarray(a).astype(np.float32) + rng.normal(0, 3.5, (H, W, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


orig = render(record=True)
plain = genish(render(dose="800 mg", batch="B.No. B2477    Exp 08/2021"))
sphota = genish(render())
coarse = orig.resize((W // 14, H // 14), Image.BILINEAR).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))
tiny = orig.resize((W // 5, H // 5), Image.LANCZOS)
import io
buf = io.BytesIO(); tiny.save(buf, "JPEG", quality=6); jpeg_bytes = buf.tell()
jpeg = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((W, H), Image.BILINEAR)


def save(im, path, q=82, size=None):
    im = im.convert("RGB")
    if size:
        im = im.resize(size, Image.LANCZOS)
    im.save(PUB / path, "WEBP", quality=q)


save(orig, "hero.webp", 80, (1000, 750))
save(orig, "proof/original.webp"); save(jpeg, "proof/jpeg.webp"); save(coarse, "proof/coarse.webp")
save(plain, "proof/plain.webp"); save(sphota, "proof/sphota.webp")

# uncertainty map (grayscale): high on invented texture, low on locked text
low = rng.random((H // 60, W // 60)).astype(np.float32)
field = np.asarray(Image.fromarray((low * 255).astype("uint8")).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(20))).astype(np.float32) / 255
field = (field - field.min()) / (field.max() - field.min())
umap = 0.04 + 0.30 * field
yy, xx = np.mgrid[0:H, 0:W]
umap[:, :] *= np.where((xx > 150) & (xx < 1050) & (yy > 100) & (yy < 780), 0.85, 1.0)
for k in ("name", "dose", "batch"):
    x0, y0, x1, y1 = boxes[k]; umap[y0:y1, x0:x1] = 0.03
umap[:130, :] = 0.10 + 0.25 * field[:130, :]
for (cx, cy, r, v) in ((870, 690, 150, 0.95), (100, 650, 130, 0.9), (930, 150, 110, 0.75)):  # the three flagged areas
    umap += v * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 1.6) ** 2)))
for k in ("name", "dose", "batch"):
    x0, y0, x1, y1 = boxes[k]; umap[y0:y1, x0:x1] = 0.03
Image.fromarray((np.clip(umap, 0, 1) * 255).astype("uint8")).resize((600, 450), Image.BILINEAR).save(PUB / "demo/uncertainty.png")

# denoise preview frames (noisy -> sharp)
sp = np.asarray(sphota).astype(np.float32)
for i, a in enumerate((0.9, 0.7, 0.45, 0.2, 0.0), start=1):
    n = rng.normal(128, 70, sp.shape).clip(0, 255)
    fr = Image.fromarray(((1 - a) * sp + a * n).clip(0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(a * 3))
    save(fr, f"demo/preview/denoise_{i}.webp", 70, (600, 450))

# ── samples and neighbour thumbnails (simple illustrations) ──
def invoice():
    im = Image.new("RGB", (W, H), (233, 224, 205)); d = ImageDraw.Draw(im)
    d.rectangle([200, 70, 1000, 840], fill=(250, 246, 236))
    d.text((250, 110), "TAX INVOICE", font=bold(64), fill=(70, 42, 28)); d.text((250, 200), "No. 2041 · 14 Nov 2026", font=reg(34), fill=(110, 80, 60))
    for i in range(6):
        d.line([250, 300 + i * 70, 950, 300 + i * 70], fill=(205, 195, 175), width=3)
        d.text((260, 312 + i * 70), ["Consulting", "Hosting", "Support", "Licence", "Training", "Travel"][i], font=reg(34), fill=(90, 60, 45))
        d.text((780, 312 + i * 70), f"{[900, 1200, 450, 1500, 400, 300][i]:,}", font=reg(34), fill=(90, 60, 45))
    d.text((620, 740), "TOTAL  4,750", font=bold(54), fill=(201, 86, 30))
    return im


def landscape():
    im = Image.new("RGB", (W, H)); d = ImageDraw.Draw(im)
    for y in range(H):
        t = y / H; d.line([(0, y), (W, y)], fill=(int(240 - 40 * t), int(196 - 90 * t), int(120 - 60 * t)))
    d.ellipse([430, 260, 770, 600], fill=(250, 226, 170))
    d.polygon([(0, 600), (260, 420), (520, 600)], fill=(110, 74, 55)); d.polygon([(380, 620), (760, 380), (1200, 640)], fill=(90, 58, 43))
    d.rectangle([0, 620, W, H], fill=(70, 44, 32))
    return im


def face():
    im = Image.new("RGB", (W, H), (110, 74, 55)); d = ImageDraw.Draw(im)
    d.ellipse([380, 150, 820, 700], fill=(232, 190, 160)); d.ellipse([480, 330, 540, 380], fill=(60, 36, 24)); d.ellipse([660, 330, 720, 380], fill=(60, 36, 24))
    d.arc([500, 450, 700, 600], 20, 160, fill=(150, 70, 60), width=10); d.pieslice([360, 100, 840, 420], 180, 360, fill=(60, 36, 24))
    return im


def whiteboard():
    im = Image.new("RGB", (W, H), (200, 200, 205)); d = ImageDraw.Draw(im)
    d.rectangle([90, 90, 1110, 810], fill=(248, 248, 244), outline=(150, 150, 150), width=8)
    d.text((170, 160), "Sprint plan", font=bold(80), fill=(40, 70, 140))
    for i, t in enumerate(["1  Encode", "2  Decode", "3  Demo day 14/12"]):
        d.text((190, 330 + i * 120), t, font=reg(70), fill=(180, 60, 50))
    return im


save(orig, "samples/medicine.webp", 80, (600, 450)); save(invoice(), "samples/invoice.webp", 80, (600, 450)); save(landscape(), "samples/landscape.webp", 80, (600, 450))
thumbs = {"invoice": invoice(), "face": face(), "whiteboard": whiteboard(), "landscape": landscape()}
for k, v in thumbs.items():
    save(v, f"demo/thumb_{k}.webp", 75, (172, 129))

# ── JSON (demo values) ──
fr = lambda b: dict(x=round(b[0] / W, 4), y=round(b[1] / H, 4), w=round((b[2] - b[0]) / W, 4), h=round((b[3] - b[1]) / H, 4))
locked = [dict(**fr(boxes["name"]), kind="text", label="Name"), dict(**fr(boxes["dose"]), kind="text", label="Dose 500 mg"),
          dict(**fr(boxes["batch"]), kind="text", label="Batch · expiry")]
(PUB / "demo/boxes.json").write_text(json.dumps({"dose": fr(boxes["dose"]), "locked": locked}))
compress = dict(job_id="demo-001", bytes=1946, original_bytes=612000, bpp=round(1946 * 8 / (W * H), 4), mode="small", recommended_mode="small",
                boxes=locked, timings=dict(detect=0.42, recommend=0.18, encode=0.96, entropy=0.54, pack=0.13), seed_url="/demo/seed.sphota")
(PUB / "demo/compress.json").write_text(json.dumps(compress))
(PUB / "demo/seed.sphota").write_bytes(b"TSI1" + bytes(1942))

flag = lambda i, n, r, s, c, ic: dict(id=i, name=n, reason=r, score=s, box=dict(**fr(c), kind="text", label=n), icon=ic)
flags = [flag("f1", "Foil pattern", "Texture generated; not in the seed", 0.71, (700, 645, 1030, 735), "pill"),
         flag("f2", "Wood grain", "Fine texture invented by the decoder", 0.64, (0, 560, 190, 740), "texture"),
         flag("f3", "Top band text", "Small print; low confidence", 0.48, (200, 118, 700, 192), "text")]
ev = []
for st, ms in (("detect", 420), ("recommend", 180), ("encode", 960), ("entropy", 540), ("pack", 130)):
    ev += [dict(stage=st, status="running", delay=250), dict(stage=st, status="done", ms=ms, delay=ms)]
for i in range(1, 6):
    ev.append(dict(stage="denoise", step=i * 5, total=25, preview_url=f"/demo/preview/denoise_{i}.webp", delay=520))
ev.append(dict(stage="done", image_url="/proof/sphota.webp", coarse_url="/proof/coarse.webp", uncertainty_url="/demo/uncertainty.png",
               consistency=0.982, flagged=flags, delay=400))
(PUB / "demo/events.json").write_text(json.dumps(ev))

dec = lambda u, l, v, k: dict(url=u, lpips=l, verdict=v, kept=k)
compare = dict(jpeg=dec("/proof/jpeg.webp", 0.58, "Blocks and blur", False), coarse=dec("/proof/coarse.webp", 0.41, "Honest but soft", True),
               plain=dec("/proof/plain.webp", 0.24, "Dose changed to 800 mg", False), sphota=dec("/proof/sphota.webp", 0.21, "Dose kept: 500 mg", True),
               metrics=dict(tfr=dict(plain=0.46, sphota=0.98), fir=dict(plain=0.52, sphota=0.91),
                            consistency=dict(plain=0.64, sphota=0.982), decode_s=dict(plain=7.8, sphota=8.4)))
(PUB / "demo/compare.json").write_text(json.dumps(compare))

N = lambda i, l, d, c, taken, leaf, chosen=False: dict(id=i, label=l, depth=d, col=c, taken=taken, leaf=leaf, chosen=chosen)
E = lambda a, b, l, t: dict(**{"from": a}, to=b, label=l, taken=t)
explain = dict(
    tree=dict(nodes=[N("n0", "text area > 4%", 0, .5, True, False), N("n1", "→ Tiny", 1, .12, False, True), N("n2", "network ≤ 3G", 1, .62, True, False),
                     N("n3", "faces > 0", 2, .36, False, False), N("n4", "small text < 12 px", 2, .8, True, False),
                     N("n5", "→ Small", 3, .7, True, True, True), N("n6", "→ Exact", 3, .94, False, True),
                     N("n7", "→ Balanced", 3, .22, False, True), N("n8", "→ Small", 3, .46, False, True)],
               edges=[E("n0", "n1", "no", False), E("n0", "n2", "yes · 6.1%", True), E("n2", "n3", "no", False), E("n2", "n4", "yes · 2G", True),
                      E("n4", "n5", "no · 14 px", True), E("n4", "n6", "yes", False), E("n3", "n7", "yes", False), E("n3", "n8", "no", False)]),
    rules=[dict(antecedents=["small_text", "low_light", "mode_tiny"], consequent="HALLUC", support=0.04, confidence=0.71, lift=2.6, fired=False),
           dict(antecedents=["small_text", "mode_tiny"], consequent="HALLUC", support=0.09, confidence=0.64, lift=2.1, fired=True, note="Changed the mode from Tiny to Small"),
           dict(antecedents=["many_faces", "mode_tiny"], consequent="HALLUC", support=0.05, confidence=0.58, lift=1.9, fired=False)],
    neighbours=[dict(thumb_url="/demo/thumb_invoice.webp", best_mode="small", name="Invoice"), dict(thumb_url="/demo/thumb_face.webp", best_mode="balanced", name="Portrait"),
                dict(thumb_url="/demo/thumb_whiteboard.webp", best_mode="small", name="Whiteboard"), dict(thumb_url="/demo/thumb_landscape.webp", best_mode="tiny", name="Landscape")],
    predicted=dict(kb=1.9, kb_std=0.3, lpips=0.22), actual=dict(kb=1.9, lpips=0.21))
(PUB / "demo/explain.json").write_text(json.dumps(explain))

# favicon + og image
(PUB / "favicon.svg").write_text(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='14' fill='#5A3A2B'/>"
    "<defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#F5E7C6'/><stop offset='.5' stop-color='#E9B66A'/>"
    "<stop offset='1' stop-color='#C9561E'/></linearGradient><clipPath id='c'><rect x='0' y='14' width='64' height='34'/></clipPath></defs>"
    "<circle cx='32' cy='46' r='26' fill='url(#g)' clip-path='url(#c)'/><rect x='0' y='30' width='64' height='3' fill='#5A3A2B'/>"
    "<rect x='0' y='37' width='64' height='4' fill='#5A3A2B'/><rect x='0' y='46' width='64' height='18' fill='#4A2F22'/></svg>")
og = Image.new("RGB", (1200, 630), (90, 58, 43)); d = ImageDraw.Draw(og)
for y in range(300, 630):
    d.line([(0, y), (1200, y)], fill=(int(110 - 40 * (y - 300) / 330), int(74 - 30 * (y - 300) / 330), int(55 - 25 * (y - 300) / 330)))
for y in range(150, 400):
    t = (y - 150) / 250; w = int((1 - ((y - 400) / 250) ** 2) ** 0.5 * 250) if y < 400 else 0
    d.line([(850 - w, y), (850 + w, y)], fill=(int(245 - 44 * t), int(231 - 117 * t), int(198 - 168 * t)))
for k in range(7):
    d.rectangle([590, 250 + k * 20, 1110, 250 + k * 20 + 2 + k * 2], fill=(90, 58, 43))
d.line([(0, 400), (1200, 400)], fill=(201, 86, 30), width=3)
d.text((70, 150), "Sphoṭa", font=ImageFont.truetype(F + "georgiaz.ttf", 110), fill=(245, 231, 198))
d.text((74, 300), "The whole picture, in a seed.", font=ImageFont.truetype(F + "georgia.ttf", 44), fill=(233, 212, 164))
og.save(PUB / "og.png")
print("jpeg demo bytes", jpeg_bytes, "| boxes", boxes)
