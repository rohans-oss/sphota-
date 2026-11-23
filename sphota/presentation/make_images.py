"""Makes the images used by build_deck_visual.py in presentation/assets/.
Real project material: Kodak photos through the project's JPEG baseline, the Step 3/4 RD figure, and the forward
noising of the project's own diffusion schedule. Plus an illustrated (not real) Malleshwaram scene.
Run with the project venv: python presentation/make_images.py"""
import io
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A = Path(__file__).with_name("assets"); A.mkdir(exist_ok=True)
FONT = "C:/Windows/Fonts/arialbd.ttf"

# ── 1. Kodak: original vs JPEG at the smallest quality (the project's own baseline) ──
kodak = Image.open(ROOT / "data/raw/kodak/kodim01.png").convert("RGB")
kodak.save(A / "kodak_original.png")
buf = io.BytesIO(); kodak.save(buf, "JPEG", quality=1); n1 = buf.tell()
Image.open(io.BytesIO(buf.getvalue())).convert("RGB").save(A / "kodak_jpeg_q1.png")
buf = io.BytesIO(); kodak.save(buf, "JPEG", quality=75); n75 = buf.tell()
bpp1 = n1 * 8 / (kodak.width * kodak.height)
(A / "numbers.txt").write_text(f"jpeg_q1_bytes={n1}\njpeg_q1_bpp={bpp1:.3f}\njpeg_q75_bytes={n75}\n"
                               f"target_0.05bpp_bytes={kodak.width * kodak.height * 0.05 / 8:.0f}\n")
print("kodak", n1, f"{bpp1:.3f} bpp", n75)

# ── 2. RD curve: PSNR panel of the project's Step 3/4 figure ──
fig = Image.open(ROOT / "report/figures/rd_kodak_all.png")
w, h = fig.size
fig.crop((0, 0, int(w * 0.335), h)).save(A / "rd_psnr.png")

# ── 3. Diffusion forward process with the project's own schedule ──
import torch  # noqa: E402
from truesight.diffusion.schedule import Schedule  # noqa: E402

crop = kodak.crop((256, 128, 512, 384))
x0 = torch.from_numpy(__import__("numpy").asarray(crop)).permute(2, 0, 1).float().div(255) * 2 - 1
sch = Schedule(1000, "cosine")
torch.manual_seed(1)
eps = torch.randn_like(x0)
tiles = []
for t in (0, 250, 500, 750, 999):
    xt = sch.q_sample(x0[None], torch.tensor([t]), eps[None])[0]
    tiles.append(Image.fromarray(((xt.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()))
strip = Image.new("RGB", (5 * 256 + 4 * 12, 256), (251, 250, 247))
for i, im in enumerate(tiles):
    strip.paste(im, (i * 268, 0))
strip.save(A / "diffusion_forward.png")

# ── 4. Malleshwaram scene (illustration) ──
W, H = 1200, 700


def font(sz):
    return ImageFont.truetype(FONT, sz)


def base_scene():
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    for y in range(H):                                              # sky
        t = y / H; d.line([(0, y), (W, y)], fill=(int(190 + 50 * t), int(222 + 25 * t), int(247 + 6 * t)))
    rnd = random.Random(7)
    x = -20
    while x < W:                                                    # far buildings
        bw, bh = rnd.randint(110, 190), rnd.randint(190, 330)
        col = rnd.choice([(236, 214, 190), (214, 226, 238), (240, 226, 200), (222, 208, 224), (210, 232, 222)])
        d.rectangle([x, 470 - bh, x + bw, 470], fill=col)
        for wy in range(470 - bh + 22, 440, 46):
            for wx in range(x + 14, x + bw - 24, 36):
                d.rectangle([wx, wy, wx + 16, wy + 24], fill=(250, 250, 245))
        x += bw + rnd.randint(-6, 10)
    for tx in (90, 520, 1010):                                      # trees
        d.rectangle([tx - 6, 400, tx + 6, 480], fill=(120, 92, 70))
        for ox, oy, r in ((0, 360, 58), (-34, 392, 44), (36, 390, 46)):
            d.ellipse([tx + ox - r, oy - r, tx + ox + r, oy + r], fill=(112, 168, 118))
    d.rectangle([0, 470, W, 520], fill=(226, 222, 214))             # footpath
    d.rectangle([0, 520, W, H], fill=(150, 154, 162))               # road
    for lx in range(30, W, 190):
        d.rectangle([lx, 630, lx + 110, 642], fill=(246, 246, 240))
    # injured person (lying on the road) and fallen bike
    d.ellipse([300, 585, 352, 637], fill=(232, 190, 160))           # head
    d.rounded_rectangle([340, 590, 500, 640], 22, fill=(214, 92, 84))  # shirt
    d.rounded_rectangle([488, 596, 640, 636], 18, fill=(56, 70, 104))  # legs
    d.ellipse([740, 600, 830, 690], outline=(40, 44, 52), width=9)  # bike wheels
    d.ellipse([880, 600, 970, 690], outline=(40, 44, 52), width=9)
    d.line([785, 645, 860, 610, 925, 645], fill=(190, 60, 60), width=8)
    return im


def draw_sign(im, text, wobble=False, seed=3):
    d = ImageDraw.Draw(im)
    d.rectangle([592, 300, 608, 520], fill=(120, 126, 136))         # pole
    d.rounded_rectangle([300, 190, 900, 310], 14, fill=(44, 104, 172), outline="white", width=6)
    f = font(66); rnd = random.Random(seed)
    tw = sum(d.textlength(c, font=f) for c in text); x = 600 - tw / 2
    for c in text:
        cw = d.textlength(c, font=f)
        if wobble:
            tile = Image.new("RGBA", (int(cw) + 40, 120), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((20, 20), c, font=f, fill="white")
            tile = tile.rotate(rnd.uniform(-9, 9), resample=Image.BICUBIC)
            im.paste(tile, (int(x) - 20, 215 + rnd.randint(-5, 5)), tile)
        else:
            d.text((x, 214), c, font=f, fill="white")
        x += cw
    return im


def smooth(im, r=3):
    s = im.filter(ImageFilter.GaussianBlur(r))
    return Image.blend(s, im.filter(ImageFilter.SMOOTH_MORE), 0.3)


orig = draw_sign(base_scene(), "MALLESHWARAM"); orig.save(A / "scene_original.png")

ai = smooth(base_scene(), 2.2); ai = draw_sign(ai, "MALLASHWORAM", wobble=True)       # redrawn text: wrong letters
ai.save(A / "scene_ai.png")

ts = smooth(base_scene(), 2.2)                                                         # generated background
sign_layer = draw_sign(base_scene(), "MALLESHWARAM")
ts.paste(sign_layer.crop((280, 170, 920, 530)), (280, 170))                            # protected: exact sign
ts.paste(orig.crop((280, 570, 660, 650)), (280, 570))                                  # protected: victim
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
for box in ((20, 90, 260, 330), (950, 60, 1180, 300), (680, 560, 1010, 700)):          # flagged: invented detail
    od.rounded_rectangle(box, 18, fill=(255, 214, 90, 120))
ts = Image.alpha_composite(ts.convert("RGBA"), ov)
dd = ImageDraw.Draw(ts)
for box in ((286, 176, 914, 322), (286, 566, 660, 654)):                               # protected boxes (dashed)
    x0, y0, x1, y1 = box
    for xx in range(x0, x1, 26):
        dd.line([(xx, y0), (min(xx + 14, x1), y0)], fill=(30, 140, 110), width=5)
        dd.line([(xx, y1), (min(xx + 14, x1), y1)], fill=(30, 140, 110), width=5)
    for yy in range(y0, y1, 26):
        dd.line([(x0, yy), (x0, min(yy + 14, y1))], fill=(30, 140, 110), width=5)
        dd.line([(x1, yy), (x1, min(yy + 14, y1))], fill=(30, 140, 110), width=5)
ts.convert("RGB").save(A / "scene_truesight.png")
print("assets:", sorted(p.name for p in A.iterdir()))
