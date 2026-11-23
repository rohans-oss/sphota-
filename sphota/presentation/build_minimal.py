"""Minimal 5-slide TrueSight deck: light background, no boxes, very little text.
Run: python presentation/build_minimal.py"""
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("TrueSight_minimal.pptx")
C = RGBColor.from_string
BG, INK, GREY, RULE = C("FBFAF7"), C("23272F"), C("8A8F98"), C("D9D6CF")
BLUE, RED, GREEN = C("3A6EA5"), C("C0453F"), C("2E8B6E")
HEAD, BODY = "Georgia", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    return s


def text(s, x, y, w, h, runs, size=20, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: str, or list of lines; each line is str or list of (str, {color,bold,size,font})."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = [runs] if isinstance(runs, str) else runs
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(6)
        for r in ([ln] if isinstance(ln, str) else ln):
            t, o = (r, {}) if isinstance(r, str) else r
            run = p.add_run(); run.text = t; f = run.font
            f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return tb


def line(s, x1, y1, x2, y2, color=RULE, w=1.0, arrow=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(w)
    if arrow:
        ln = c.line._get_or_add_ln()
        tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
        tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return c


def title(s, t, sub=None):
    text(s, 0.9, 0.7, 11.5, 0.9, t, size=40, font=HEAD)
    if sub:
        text(s, 0.9, 1.65, 11.5, 0.5, sub, size=20, color=GREY)


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


# 1 ── problem ───────────────────────────────────────────────────────────────
s = slide()
title(s, "Tiny photos that lie")
text(s, 0.9, 2.3, 5.4, 3.2, ["AI can shrink a photo 20× and make it look perfect.",
                             [("But it can quietly change what the photo says.", {"color": RED})]], size=26, font=HEAD)
text(s, 7.3, 2.0, 2.4, 0.4, "You send", size=14, color=GREY)
text(s, 10.2, 2.0, 2.6, 0.4, "AI delivers", size=14, color=GREY)
rows = [("5 mg", "8 mg"), ("₹4,750", "₹4,150"), ("your child", "someone else"), ("a plate", "another plate")]
for i, (a, b) in enumerate(rows):
    y = 2.55 + i * 0.95
    line(s, 7.3, y - 0.12, 12.4, y - 0.12)
    text(s, 7.3, y, 2.7, 0.6, a, size=28)
    line(s, 9.75, y + 0.3, 10.1, y + 0.3, color=GREY, arrow=True)
    text(s, 10.2, y, 2.6, 0.6, b, size=28, color=RED)
line(s, 7.3, 2.55 + 4 * 0.95 - 0.12, 12.4, 2.55 + 4 * 0.95 - 0.12)
notes(s, "Generative codecs look best at tiny sizes but can silently change letters, numbers and faces. That blocks real use.")

# 2 ── solution ──────────────────────────────────────────────────────────────
s = slide()
title(s, "TrueSight", "Send small. Look real. Never change what matters.")
steps = [("Photo", "any phone"), ("Tiny file", "text and faces kept sharp"), ("Rebuild", "AI adds texture"),
         ("Photo + flags", "invented parts marked")]
xs = [0.9, 4.0, 7.1, 9.95]
for (t, d), x in zip(steps, xs):
    text(s, x, 3.5, 3.0, 0.7, t, size=30, font=HEAD)
    text(s, x, 4.2, 2.6, 0.6, d, size=15, color=GREY)
for x in (3.2, 6.3, 9.0):
    line(s, x, 3.82, x + 0.55, 3.82, color=GREY, arrow=True)
line(s, 0.9, 5.35, 12.4, 5.35)
text(s, 0.9, 5.65, 3.5, 0.7, "Small", size=32, font=HEAD, color=BLUE)
text(s, 4.9, 5.65, 3.5, 0.7, "Beautiful", size=32, font=HEAD, color=GREEN)
text(s, 8.9, 5.65, 3.5, 0.7, "Faithful", size=32, font=HEAD, color=RED)
notes(s, "Sender: photo -> tiny file (text and faces kept sharp). Receiver: rebuild with diffusion, then flag whatever was invented.")

# 3 ── Malleshwaram ──────────────────────────────────────────────────────────
s = slide()
title(s, "An accident in Malleshwaram", "A bystander sends a photo. The police must read the sign.")
cols = [("What was shot", "MALLESHWARAM", BLUE, "✓  exact", GREEN),
        ("Typical AI codec", "MALLASHWORAM", BLUE, "✗  redrawn: wrong place", RED),
        ("TrueSight", "MALLESHWARAM", BLUE, "✓  exact and flagged", GREEN)]
for i, (lab, sign, col, res, rc) in enumerate(cols):
    x = 0.9 + i * 4.0
    text(s, x, 2.75, 3.4, 0.4, lab, size=15, color=GREY)
    sg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(3.2), Inches(3.3), Inches(0.95))
    sg.shadow.inherit = False; sg.fill.solid(); sg.fill.fore_color.rgb = col; sg.line.fill.background()
    tf = sg.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = sign; r.font.name = BODY; r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = C("FFFFFF")
    line(s, x + 1.65, 4.15, x + 1.65, 5.0, color=GREY, w=3)
    text(s, x, 5.25, 3.5, 0.5, res, size=20, color=rc)
line(s, 0.9, 5.0, 12.4, 5.0, color=RULE, w=2)
text(s, 0.9, 6.3, 11.8, 0.6, "TrueSight sends the sign exact, and flags whatever the AI invented.", size=24, font=HEAD)
notes(s, "Illustrative: the wrong spelling is made up to show the failure. TrueSight detects the sign text and sends it as a protected region.")

# 4 ── concepts 1 ────────────────────────────────────────────────────────────
s = slide()
title(s, "Making the file small")
flow = [("Photo", ""), ("Numbers", "encoder (VAE)"), ("Bits", "probability model (GMM)")]
fx = [0.9, 4.6, 8.8]
for (t, d), x in zip(flow, fx):
    text(s, x, 2.3, 3.2, 0.8, t, size=38, font=HEAD)
    text(s, x, 3.1, 3.4, 0.5, d, size=15, color=GREY)
for x in (2.5, 7.0):
    line(s, x, 2.75, x + 1.2, 2.75, color=GREY, arrow=True)
line(s, 0.9, 4.2, 12.4, 4.2)
facts = [("VAE", "photo → small integers"), ("GMM + MLE", "cost of a number = −log₂ p"), ("λ", "bits vs. quality: six modes")]
for i, (h, b) in enumerate(facts):
    x = 0.9 + i * 4.0
    text(s, x, 4.6, 3.6, 0.6, h, size=26, font=HEAD, color=BLUE)
    text(s, x, 5.3, 3.6, 0.8, b, size=19)
notes(s, "A VAE turns the image into integers. A Gaussian-mixture model, trained by maximum likelihood, says how many bits each integer costs. Lambda trades bits against error.")

# 5 ── concepts 2 ────────────────────────────────────────────────────────────
s = slide()
title(s, "Making it real, and true")
parts = [("Diffusion", BLUE, ["Adds realistic texture", "starting from the blurry decode"]),
         ("Guidance", GREEN, ["Matches the sent bits", "text and faces stay exact"]),
         ("Warnings", RED, ["Flags what the AI invented", "safe mode shows the blurry truth"])]
for i, (h, col, ls) in enumerate(parts):
    x = 0.9 + i * 4.1
    text(s, x, 2.9, 3.6, 0.7, h, size=34, font=HEAD, color=col)
    text(s, x, 3.9, 3.5, 1.6, [[(ls[0], {"size": 21})], [(ls[1], {"size": 17, "color": GREY})]])
    if i:
        line(s, x - 0.35, 2.9, x - 0.35, 5.3)
notes(s, "Diffusion adds texture. Bin-consistency and ROI-lock guidance keep the output faithful to the transmitted bits. A hallucination map flags invented regions.")

prs.save(OUT)
print("saved", OUT)
