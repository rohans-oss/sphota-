"""TrueSight 5-slide deck, light theme, short text + real project images. Run make_images.py first.
Run: python presentation/build_deck_visual.py"""
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
A = HERE / "assets"
OUT = HERE / "TrueSight_visual.pptx"
C = RGBColor.from_string
BG, INK, GREY, RULE = C("FBFAF7"), C("23272F"), C("7C828C"), C("D9D6CF")
BLUE, RED, GREEN = C("3A6EA5"), C("C0453F"), C("2E8B6E")
HEAD, BODY = "Georgia", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def slide():
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    return s


def text(s, x, y, w, h, lines, size=20, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, ln in enumerate([lines] if isinstance(lines, str) else lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(5)
        for r in ([ln] if isinstance(ln, str) else ln):
            t, o = (r, {}) if isinstance(r, str) else r
            run = p.add_run(); run.text = t; f = run.font
            f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)


def line(s, x1, y1, x2, y2, color=RULE, w=1.0, arrow=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(w)
    if arrow:
        t = etree.SubElement(c.line._get_or_add_ln(), "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
        t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")


def pic(s, name, x, y, w):
    p = s.shapes.add_picture(str(A / name), Inches(x), Inches(y), width=Inches(w))
    p.line.color.rgb = RULE; p.line.width = Pt(0.75)
    return p


def title(s, t, sub=None):
    text(s, 0.9, 0.55, 11.5, 0.8, t, size=36, font=HEAD)
    if sub:
        text(s, 0.9, 1.35, 11.5, 0.4, sub, size=18, color=GREY)


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


# 1 ── problem ───────────────────────────────────────────────────────────────
s = slide()
title(s, "The problem: tiny photos fail", "Two ways: classical codecs blur, AI codecs invent.")
pic(s, "kodak_original.png", 0.9, 2.2, 3.7); pic(s, "kodak_jpeg_q1.png", 4.85, 2.2, 3.7)
text(s, 0.9, 4.8, 3.7, 0.7, "Original photo", size=15, color=GREY)
text(s, 4.85, 4.8, 3.7, 0.9, ["JPEG at its smallest setting", [("9 KB instead of 92 KB: blocks and blur", {"color": RED})]], size=15, color=GREY)
text(s, 9.1, 2.15, 3.6, 0.5, "AI codecs fix the look, but can change the content:", size=17, color=GREY)
for i, (a, b) in enumerate([("5 mg", "8 mg"), ("₹4,750", "₹4,150"), ("a face", "another face"), ("a plate", "another plate")]):
    y = 3.0 + i * 0.72
    line(s, 9.1, y - 0.08, 12.5, y - 0.08)
    text(s, 9.1, y, 1.7, 0.5, a, size=22)
    line(s, 10.75, y + 0.22, 11.1, y + 0.22, color=GREY, arrow=True)
    text(s, 11.2, y, 1.9, 0.5, b, size=22, color=RED)
line(s, 9.1, 3.0 + 4 * 0.72 - 0.08, 12.5, 3.0 + 4 * 0.72 - 0.08)
line(s, 0.9, 5.95, 12.5, 5.95)
text(s, 0.9, 6.2, 11.6, 0.6, "Goal: tiny files that still tell the truth.", size=24, font=HEAD)
notes(s, "Left: a real Kodak photo through our own JPEG baseline at quality 1 (9,383 bytes vs 92,491 bytes at quality 75). "
         "Right: generative decoders look great but can rewrite letters, numbers and faces.")

# 2 ── solution ──────────────────────────────────────────────────────────────
s = slide()
title(s, "The solution: TrueSight", "Send small. Look real. Never change what matters.")
pic(s, "scene_original.png", 0.9, 2.3, 3.6)
text(s, 0.9, 4.45, 3.6, 0.5, "Photo in", size=16, color=GREY)
line(s, 4.65, 3.3, 5.25, 3.3, color=GREY, w=1.5, arrow=True)
text(s, 5.4, 2.55, 2.5, 0.8, "10–30×", size=38, font=HEAD, color=BLUE)
text(s, 5.4, 3.35, 2.5, 1.2, ["smaller file", "text and faces", "kept sharp"], size=16, color=GREY)
line(s, 7.95, 3.3, 8.55, 3.3, color=GREY, w=1.5, arrow=True)
pic(s, "scene_truesight.png", 8.7, 2.3, 3.6)
text(s, 8.7, 4.45, 3.7, 0.5, "Photo out + warnings", size=16, color=GREY)
line(s, 0.9, 5.3, 12.4, 5.3)
text(s, 0.9, 5.55, 3.6, 0.6, "Small", size=30, font=HEAD, color=BLUE)
text(s, 0.9, 6.15, 3.6, 0.5, "VAE codec, entropy-coded bits", size=15, color=GREY)
text(s, 4.9, 5.55, 3.6, 0.6, "Beautiful", size=30, font=HEAD, color=GREEN)
text(s, 4.9, 6.15, 3.6, 0.5, "diffusion decoder adds texture", size=15, color=GREY)
text(s, 8.9, 5.55, 3.6, 0.6, "Faithful", size=30, font=HEAD, color=RED)
text(s, 8.9, 6.15, 3.6, 0.5, "guards + a warning map", size=15, color=GREY)
notes(s, "10-30x is the guide's target versus JPEG, not a measured result yet. The two images are an illustration of the Malleshwaram case on the next slide.")

# 3 ── Malleshwaram ──────────────────────────────────────────────────────────
s = slide()
title(s, "An accident in Malleshwaram", "A bystander sends a photo. The police must read the sign.")
cols = [("scene_original.png", "What was shot", "✓  sign is exact", GREEN, ""),
        ("scene_ai.png", "Typical AI codec", "✗  sign redrawn: wrong place", RED, ""),
        ("scene_truesight.png", "TrueSight", "✓  sign exact, invented parts flagged", GREEN, "")]
for i, (img, lab, res, rc, _) in enumerate(cols):
    x = 0.9 + i * 4.05
    text(s, x, 2.0, 3.7, 0.4, lab, size=16, color=GREY)
    pic(s, img, x, 2.4, 3.7)
    text(s, x, 4.85, 3.8, 0.5, res, size=18, color=rc)
text(s, 8.95, 5.35, 3.8, 0.9, ["dashed green = protected (sign, victim)", "yellow = flagged: AI invented this"], size=13, color=GREY)
line(s, 0.9, 5.95, 12.5, 5.95)
text(s, 0.9, 6.2, 11.6, 0.6, "TrueSight: the police read “Malleshwaram” correctly.", size=24, font=HEAD)
notes(s, "This scene is an illustration, not a real photo, and the wrong spelling is made up to show the failure. "
         "In the real system the sign text and the face are detected and sent as protected regions, the rest is generated, "
         "and a hallucination map flags invented areas.")

# 4 ── concepts 1 ────────────────────────────────────────────────────────────
s = slide()
title(s, "Concept 1: making the file small")
for (t, x) in (("Photo", 0.9), ("Numbers", 3.15), ("Bits", 5.9)):
    text(s, x, 1.65, 2.0, 0.6, t, size=30, font=HEAD)
line(s, 2.3, 1.97, 3.0, 1.97, color=GREY, arrow=True); line(s, 5.1, 1.97, 5.75, 1.97, color=GREY, arrow=True)
facts = [("VAE", BLUE, "turns the photo into small integers"),
         ("GMM", GREEN, "gives each integer a probability: likely numbers cost fewer bits  (bits = −log₂ p)"),
         ("λ", RED, "one knob trades bits against quality: six modes")]
for i, (h, col, b) in enumerate(facts):
    y = 2.9 + i * 1.2
    line(s, 0.9, y - 0.12, 6.4, y - 0.12)
    text(s, 0.9, y, 1.3, 0.6, h, size=26, font=HEAD, color=col)
    text(s, 2.2, y + 0.05, 4.3, 1.0, b, size=17)
pic(s, "rd_psnr.png", 7.3, 1.75, 5.0)
text(s, 7.3, 5.95, 5.0, 0.8, ["Our Kodak results so far (PSNR vs bits per pixel)", "JPEG, WebP, PCA, VQ: the baselines the VAE codec must beat"], size=13, color=GREY)
notes(s, "Right: the real Step 3/4 rate-distortion plot from our experiments. Lower bits per pixel means lower quality; lambda chooses the point on that curve.")

# 5 ── concepts 2 ────────────────────────────────────────────────────────────
s = slide()
title(s, "Concept 2: realistic and true", "Diffusion removes noise step by step to rebuild the photo.")
pic(s, "diffusion_forward.png", 0.9, 2.0, 11.5)
tw = 11.5 * 256 / 1328
for i, lab in enumerate(["photo  t=0", "t=250", "t=500", "t=750", "pure noise  t=999"]):
    text(s, 0.9 + i * 11.5 * 268 / 1328, 4.3, 2.4, 0.4, lab, size=13, color=GREY)
text(s, 0.9, 4.65, 11.5, 0.4, "Training adds noise like this (left to right); decoding runs it backwards (right to left).", size=14, color=GREY)
line(s, 0.9, 5.2, 12.4, 5.2)
parts = [("Conditioned", BLUE, "starts from the blurry exact decode"),
         ("Guided", GREEN, "output must match the sent bits; text and faces locked"),
         ("Warned", RED, "a map flags what the AI invented")]
for i, (h, col, b) in enumerate(parts):
    x = 0.9 + i * 4.0
    text(s, x, 5.4, 3.6, 0.6, h, size=26, font=HEAD, color=col)
    text(s, x, 6.0, 3.6, 0.9, b, size=16)
notes(s, "The strip is made with our own cosine noise schedule (truesight/diffusion/schedule.py) on a Kodak crop. "
         "Bin-consistency and ROI-lock guidance keep the output faithful; the hallucination map flags invented regions.")

prs.save(OUT)
print("saved", OUT)
