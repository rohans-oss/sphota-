"""Builds TrueSight_5_slides.pptx (light theme, 16:9). Run: python presentation/build_deck.py"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("TrueSight_5_slides.pptx")

C = lambda h: RGBColor.from_string(h)
BG, WHITE, INK, MUTED, LINE = C("F6F9FC"), C("FFFFFF"), C("1E2B45"), C("5B6B84"), C("DCE5F0")
THEMES = {  # (soft fill, accent)
    "blue": (C("DDEBFA"), C("2F6DB5")), "teal": (C("DDF3EE"), C("1E8A78")), "amber": (C("FFF0D2"), C("B87608")),
    "rose": (C("FCE3E6"), C("C0404F")), "lav": (C("E9E6FA"), C("5E4FBA")), "grey": (C("EEF2F7"), C("5B6B84")),
}
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def box(sl, x, y, w, h, fill=WHITE, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, lw=1.0, dash=False):
    s = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(lw)
        if dash:
            from pptx.enum.dml import MSO_LINE
            s.line.dash_style = MSO_LINE.DASH
    return s


def text(sl, x, y, w, h, paras, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         shape=None, margin=0.05, space=2):
    """paras: str | list of str | list of (str, {size,bold,color,italic}) ; a str may hold several runs via list of tuples."""
    tb = shape or sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = Inches(0.03)
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(space)
        runs = p if isinstance(p, list) else [p]
        for r in runs:
            t, o = (r, {}) if isinstance(r, str) else r
            run = para.add_run(); run.text = t
            f = run.font; f.name = FONT; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False); f.color.rgb = o.get("color", color)
    return tb


def card(sl, x, y, w, h, theme, title, body, tsize=14, bsize=11.5, fill=None, title_color=None):
    soft, acc = THEMES[theme]
    box(sl, x, y, w, h, fill=fill or WHITE, line=LINE)
    box(sl, x, y + 0.08, 0.07, h - 0.16, fill=acc, line=None, radius=0.5)             # accent bar
    paras = [[(title, {"size": tsize, "bold": True, "color": title_color or acc})]]
    paras += [[(b, {"size": bsize, "color": INK})] if isinstance(b, str) else b for b in body]
    text(sl, x + 0.18, y + 0.06, w - 0.28, h - 0.1, paras, space=3)


def pill(sl, x, y, w, h, label, theme, size=12, bold=True):
    soft, acc = THEMES[theme]
    s = box(sl, x, y, w, h, fill=soft, line=None, radius=0.5)
    text(sl, 0, 0, 0, 0, [[(label, {"size": size, "bold": bold, "color": acc})]], shape=s, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, margin=0.08)


def arrow(sl, x, y, w=0.32, h=0.26, color=MUTED, shape=MSO_SHAPE.RIGHT_ARROW):
    s = box(sl, x, y, w, h, fill=color, line=None, shape=shape)
    return s


def header(sl, n, title, sub):
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb = BG
    box(sl, 0, 0, 13.333, 0.12, fill=THEMES["blue"][1], line=None, shape=MSO_SHAPE.RECTANGLE)
    text(sl, 0.6, 0.3, 11.2, 0.7, [[(title, {"size": 30, "bold": True, "color": INK})]])
    text(sl, 0.6, 0.98, 11.8, 0.5, [[(sub, {"size": 15, "color": MUTED})]])
    pill(sl, 11.95, 0.4, 0.8, 0.36, f"{n} / 5", "blue", size=12)
    text(sl, 0.6, 7.08, 8, 0.3, [[("TrueSight (code name Sphoṭa) · a faithful generative image codec", {"size": 10, "color": MUTED})]])


def notes(sl, s):
    sl.notes_slide.notes_text_frame.text = s


# ───────────────────────────── Slide 1: problem ─────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, 1, "The problem: tiny photos that quietly lie",
       "AI codecs can shrink a photo ~20× and make it look perfect, but they invent details that were never sent.")

stats = [("≈ 6 billion", "photos are taken every day, and images + video are most mobile traffic", "blue"),
         ("≈ ¾", "of global mobile data is video, so every byte saved matters", "teal"),
         ("3.1 billion", "people live under mobile coverage but stay offline; data cost is a main barrier", "amber")]
for i, (big, small, th) in enumerate(stats):
    y = 1.75 + i * 1.28
    soft, acc = THEMES[th]
    box(s, 0.6, y, 3.7, 1.14, fill=soft, line=None)
    text(s, 0.75, y + 0.06, 3.45, 0.5, [[(big, {"size": 24, "bold": True, "color": acc})]])
    text(s, 0.75, y + 0.55, 3.45, 0.6, [[(small, {"size": 11, "color": INK})]])
text(s, 0.6, 5.58, 3.7, 0.3, [[("Sources cited in the build guide §2.1: Photutorial, Ericsson, GSMA 2025", {"size": 8.5, "color": MUTED, "italic": True})]])

# three generations of codec
text(s, 4.6, 1.7, 8, 0.3, [[("Why today's codecs are not enough", {"size": 14, "bold": True, "color": INK})]])
gens = [("JPEG / WebP", "Below ~0.1 bits per pixel the image turns into blocks and blur. Honest, but unusable.", "grey"),
        ("Learned codecs (VAE)", "Beat JPEG, but at extreme compression output the average image: blurry, plastic.", "lav"),
        ("Generative codecs", "Look excellent at 0.01 to 0.05 bpp, but the decoder hallucinates.", "rose")]
for i, (t, b, th) in enumerate(gens):
    x = 4.6 + i * 2.75
    card(s, x, 2.05, 2.6, 1.3, th, t, [b], tsize=13, bsize=10.5)
    if i < 2:
        arrow(s, x + 2.6, 2.6, 0.15, 0.2, color=MUTED)

# table
text(s, 4.6, 3.5, 8, 0.3, [[("What the user sent  →  what a hallucinating codec can deliver", {"size": 14, "bold": True, "color": INK})]])
rows = [("Medicine label “5 mg”", "Sharp label reading “8 mg”", "Health risk"),
        ("Invoice total “4,750”", "Clean digits “4,150”", "Financial dispute"),
        ("Photo of your child", "A realistic but different face", "Loss of trust"),
        ("Dashcam licence plate", "A plausible, wrong plate", "Evidence worthless")]
x0, ws = 4.6, [2.85, 3.2, 2.05]
y = 3.85
for j, h in enumerate(["User sent", "Codec delivers", "Consequence"]):
    xx = x0 + sum(ws[:j])
    b = box(s, xx, y, ws[j] - 0.04, 0.34, fill=THEMES["blue"][1], line=None, radius=0.15)
    text(s, 0, 0, 0, 0, [[(h, {"size": 11, "bold": True, "color": WHITE})]], shape=b, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)
for i, r in enumerate(rows):
    yy = y + 0.4 + i * 0.4
    for j, v in enumerate(r):
        xx = x0 + sum(ws[:j])
        b = box(s, xx, yy, ws[j] - 0.04, 0.36, fill=WHITE if j < 2 else THEMES["rose"][0], line=LINE, radius=0.15)
        text(s, 0, 0, 0, 0, [[(v, {"size": 11, "color": THEMES["rose"][1] if j == 2 else INK, "bold": j == 2})]],
             shape=b, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)

# formal problem statement
b = box(s, 0.6, 5.95, 12.13, 1.0, fill=WHITE, line=THEMES["blue"][1], lw=1.5)
text(s, 0.8, 6.0, 11.8, 0.95, [
    [("Problem statement.  ", {"size": 13, "bold": True, "color": THEMES["blue"][1]}),
     ("Given an image x and a bit budget R, produce a bitstream b (|b| ≤ R) and a decoder D so that x̂ = D(b) is ", {"size": 13}),
     ("realistic", {"size": 13, "bold": True}), (", ", {"size": 13}), ("faithful", {"size": 13, "bold": True}),
     (" (text and faces preserved), ", {"size": 13}), ("consistent", {"size": 13, "bold": True}),
     (" (re-encoding x̂ gives the transmitted latent) and ", {"size": 13}), ("honest", {"size": 13, "bold": True}),
     (" (invented regions are flagged).", {"size": 13})]], anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Open with the contradiction: generative codecs look best at tiny sizes but can silently change letters, numbers and faces. "
         "That single flaw blocks real use. The formal statement has four requirements: realistic, faithful, consistent, honest.")

# ───────────────────────────── Slide 2: solution ─────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, 2, "The solution: TrueSight, a faithful generative codec",
       "A tiny VAE code + a diffusion decoder for beauty + guards that keep text and faces exact and flag anything invented.")

promises = [("Small", "Targets 10–30× smaller files than JPEG at similar visual quality (about 0.03 to 0.06 bpp).", "blue"),
            ("Beautiful", "Conditional diffusion decoder restores realistic texture instead of blur.", "teal"),
            ("Faithful and honest", "Text and faces are sent protected; every invented region is flagged, with a safe fallback.", "amber")]
for i, (t, b, th) in enumerate(promises):
    card(s, 0.6 + i * 4.1, 1.65, 3.95, 1.05, th, t, [b], tsize=15, bsize=11)

def flow(sl, y, label, theme, items):
    soft, acc = THEMES[theme]
    pill(sl, 0.6, y + 0.18, 1.35, 0.5, label, theme, size=12)
    n = len(items); w = 2.3; gap = 0.4; x = 2.15
    for i, (t, d) in enumerate(items):
        b = box(sl, x, y, w, 0.88, fill=WHITE, line=acc, lw=1.25)
        text(sl, 0, 0, 0, 0, [[(t, {"size": 12, "bold": True, "color": acc})], [(d, {"size": 10, "color": INK})]],
             shape=b, anchor=MSO_ANCHOR.MIDDLE, margin=0.1, space=1)
        if i < n - 1:
            arrow(sl, x + w + 0.07, y + 0.31, gap - 0.14, 0.26, color=acc)
        x += w + gap

flow(s, 2.95, "SENDER", "blue", [
    ("1  Photo in", "Any photo from a phone, CCTV or dashcam"),
    ("2  Protect", "Detect text and faces; send them as sharp crops"),
    ("3  VAE codec + GMM", "Image → small integer latent, arithmetic-coded"),
    ("4  Tiny .tsi file", "Header + latent bits + protected crops + seed")])
flow(s, 4.05, "RECEIVER", "teal", [
    ("5  Honest coarse decode", "Exact decode of the latent: blurry but true"),
    ("6  Diffusion decoder", "Adds realistic texture, conditioned on step 5"),
    ("7  Guidance", "Bin-consistency + ROI-lock keep it faithful"),
    ("8  Photo + uncertainty map", "Invented regions flagged; safe mode available")])

text(s, 0.6, 5.12, 8, 0.3, [[("Five building blocks", {"size": 14, "bold": True, "color": INK})]])
blocks = [("1  VAE codec + GMM", "Rate = −log₂ p(latent): the bit cost is a learned likelihood.", "blue"),
          ("2  Diffusion decoder", "Realistic texture; guidance scale w trades realism for fidelity.", "teal"),
          ("3  Faithfulness layer", "Text/face ROI coding + bin-consistency guidance.", "amber"),
          ("4  Honesty layer", "Hallucination map, risk model and safe fallback.", "rose"),
          ("5  Intelligence layer", "Rate control, mode recommender, failure rules.", "lav")]
for i, (t, d, th) in enumerate(blocks):
    soft, acc = THEMES[th]
    b = box(s, 0.6 + i * 2.45, 5.5, 2.33, 1.4, fill=soft, line=None)
    text(s, 0.7 + i * 2.45, 5.58, 2.15, 1.3, [[(t, {"size": 13.5, "bold": True, "color": acc})], [(d, {"size": 12, "color": INK})]], space=4)
notes(s, "Walk the sender row then the receiver row. The key novelty is bin-consistency guidance: any decoded image whose re-encoded latent "
         "rounds to the transmitted integers is, by definition, consistent with the file. Targets on this slide come from the build guide; they are targets, not measured results.")

# ───────────────────────────── Slide 3: Malleshwaram ─────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, 3, "A real-world case: the accident at Malleshwaram",
       "Same photo, same tiny file size. The only difference is whether the decoder is allowed to invent the sign.")

steps = [("1", "An accident in Malleshwaram", "A person is injured on a road in Malleshwaram, Bengaluru. Every minute matters.", "rose"),
         ("2", "A bystander sends a photo", "The photo shows the victim and the “MALLESHWARAM” signboard. On a weak network it must be sent tiny.", "amber"),
         ("3", "The police act on what they see", "They read the sign to know the location and dispatch an ambulance to that exact place.", "blue")]
for i, (n, t, b, th) in enumerate(steps):
    soft, acc = THEMES[th]
    x = 0.6 + i * 4.1
    box(s, x, 1.55, 3.95, 1.1, fill=soft, line=None)
    o = box(s, x + 0.12, 1.67, 0.42, 0.42, fill=acc, line=None, shape=MSO_SHAPE.OVAL)
    text(s, 0, 0, 0, 0, [[(n, {"size": 14, "bold": True, "color": WHITE})]], shape=o, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0)
    text(s, x + 0.62, 1.58, 3.25, 0.4, [[(t, {"size": 13, "bold": True, "color": acc})]])
    text(s, x + 0.62, 1.92, 3.25, 0.75, [[(b, {"size": 10.5, "color": INK})]])
    if i < 2:
        arrow(s, x + 3.97, 2.0, 0.12, 0.2, color=MUTED)


def scene(sl, x, y, w, h, sign, sign_theme, roi=False, haze=False):
    box(sl, x, y, w, h, fill=C("EAF3FB"), line=LINE)                                           # sky
    box(sl, x + 0.02, y + h * 0.62, w - 0.04, h * 0.36, fill=C("DCDFE5"), line=None, shape=MSO_SHAPE.RECTANGLE)   # road
    for k in range(4):
        box(sl, x + 0.3 + k * 0.9, y + h * 0.8, 0.5, 0.05, fill=WHITE, line=None, shape=MSO_SHAPE.RECTANGLE)    # lane marks
    box(sl, x + w / 2 - 0.04, y + 0.72, 0.08, h * 0.36, fill=C("9AA7B8"), line=None, shape=MSO_SHAPE.RECTANGLE)  # pole
    soft, acc = THEMES[sign_theme]
    sg = box(sl, x + w / 2 - 1.3, y + 0.2, 2.6, 0.58, fill=C("2F6DB5"), line=WHITE, lw=2, radius=0.12)
    text(sl, 0, 0, 0, 0, [[(sign, {"size": 17, "bold": True, "color": WHITE})]], shape=sg, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, margin=0.02)
    box(sl, x + 0.35, y + h * 0.7, 0.9, 0.34, fill=C("F2B8B5"), line=None, radius=0.5)          # injured person (body)
    box(sl, x + 1.2, y + h * 0.7 - 0.02, 0.32, 0.32, fill=C("E8C4A8"), line=None, shape=MSO_SHAPE.OVAL)  # head
    box(sl, x + w - 1.45, y + h * 0.66, 1.2, 0.46, fill=C("A9B8CC"), line=None, radius=0.2)     # vehicle
    box(sl, x + w - 1.32, y + h * 0.78, 0.3, 0.3, fill=C("6B7A90"), line=None, shape=MSO_SHAPE.OVAL)
    box(sl, x + w - 0.62, y + h * 0.78, 0.3, 0.3, fill=C("6B7A90"), line=None, shape=MSO_SHAPE.OVAL)
    if roi:
        box(sl, x + w / 2 - 1.4, y + 0.14, 2.8, 0.7, fill=None, line=THEMES["teal"][1], lw=2, dash=True, radius=0.1)
        box(sl, x + 0.25, y + h * 0.64, 1.4, 0.48, fill=None, line=THEMES["teal"][1], lw=2, dash=True, radius=0.1)
    if haze:
        for (hx, hy, hw, hh) in [(0.3, 0.93, 0.9, 0.27), (w - 1.2, 0.93, 0.9, 0.27), (1.75, h * 0.64, 0.8, 0.28)]:
            box(sl, x + hx, y + hy, hw, hh, fill=C("FFE29A"), line=None, radius=0.3)


panels = [("What the bystander shot", "grey", "MALLESHWARAM", "blue", False, False,
           "The original. Faces and the sign are all exact.", "ORIGINAL"),
          ("Plain generative codec", "rose", "MALLASHWORAM", "rose", False, False,
           "Sharp and pretty, but the AI redrew the letters. Police read the wrong place. (Illustrative spelling.)", "HALLUCINATED"),
          ("TrueSight", "teal", "MALLESHWARAM", "teal", True, True,
           "Sign and victim sent as protected crops; the rest is generated. Yellow = invented, flagged.", "FAITHFUL")]
for i, (t, th, sign, sth, roi, haze, cap, tag) in enumerate(panels):
    x = 0.6 + i * 4.1
    soft, acc = THEMES[th]
    pill(s, x, 2.82, 2.4, 0.32, t, th, size=12)
    pill(s, x + 2.55, 2.82, 1.4, 0.32, tag, th, size=9.5)
    scene(s, x, 3.22, 3.95, 1.98, sign, sth, roi=roi, haze=haze)
    text(s, x, 5.2, 3.95, 0.62, [[(cap, {"size": 10.5, "color": INK})]])

b = box(s, 0.6, 5.88, 12.13, 1.1, fill=WHITE, line=THEMES["teal"][1], lw=1.5)
text(s, 0.78, 5.9, 11.85, 1.06, [
    [("How TrueSight handles it:  ", {"size": 12.5, "bold": True, "color": THEMES["teal"][1]}),
     ("① the encoder detects the sign text and the victim's face, ② sends them as small high-quality crops, "
      "③ the diffusion decoder fills the background but ROI-lock and bin-consistency guidance keep the sign exact, "
      "④ a hallucination map flags anything invented so the police can switch to safe mode (no generation).", {"size": 12.5})],
    [("Result: ", {"size": 12.5, "bold": True, "color": THEMES["teal"][1]}),
     ("the police read “Malleshwaram” correctly and the ambulance goes to the right place.", {"size": 12.5, "bold": True})]],
     anchor=MSO_ANCHOR.MIDDLE, space=3)
notes(s, "This is a scripted illustration, not a measured result: the spelling shown on the hallucinated panel is made up to show the failure mode. "
         "In the real system the Text Faithfulness Rate (OCR character error rate <= 0.05 on text crops) measures exactly this, "
         "and the target is at least 2x that of plain diffusion decoding.")

# ───────────────────────────── Slide 4: concepts 1 ─────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, 4, "Concepts (1 of 2): making the file small",
       "A VAE turns the image into integers; a learned probability model says how many bits each integer costs.")

# diagram strip
chain = [("x", "photo"), ("g_a", "encoder CNN"), ("y → ŷ", "round to integers"), ("p(ŷ)", "GMM from hyperprior"),
         ("arithmetic coder", "torchac, real bytes"), ("bits", ".tsi file")]
x = 0.6
for i, (t, d) in enumerate(chain):
    th = ["grey", "blue", "blue", "lav", "amber", "teal"][i]
    soft, acc = THEMES[th]
    w = 1.6 if i != 4 else 2.0
    b = box(s, x, 1.65, w, 0.85, fill=soft, line=None)
    text(s, 0, 0, 0, 0, [[(t, {"size": 14, "bold": True, "color": acc})], [(d, {"size": 10, "color": INK})]], shape=b,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=1)
    if i < len(chain) - 1:
        arrow(s, x + w + 0.04, 1.97, 0.22, 0.22, color=MUTED)
    x += w + 0.28
text(s, 0.6, 2.55, 12.1, 0.3, [[("Decoder: bits → ŷ → g_s → coarse image x_c (blurry but exact). Hyperprior z = h_a(y) is coded first and predicts the GMM for y.",
                                  {"size": 11, "color": MUTED, "italic": True})]])

cards = [
    ("Bits per pixel and rate–distortion", "blue",
     ["bpp = file bits ÷ (H × W).", "Loss = bpp + λ · 255² · MSE. Six λ values (0.0018 … 0.0483) give six quality modes."]),
    ("VAE and autoencoder", "lav",
     ["Conv encoder/decoder with GDN activations; uniform noise stands in for rounding while training.",
      "A plain AE is the baseline: its bits cannot be controlled, so we need a probabilistic latent."]),
    ("Hyperprior + Gaussian mixture (K = 3)", "teal",
     ["p(ŷ) = Σₖ wₖ [Φ((ŷ+½−μₖ)/σₖ) − Φ((ŷ−½−μₖ)/σₖ)] for every latent element.",
      "Ablation: K = 1 vs K = 3."]),
    ("Maximum likelihood = fewer bits", "amber",
     ["Rate R = −Σ log₂ p(ŷ) − Σ log₂ p(ẑ).", "Minimising the bit cost is maximising the likelihood of the latents under the model."]),
    ("Exact arithmetic coding", "rose",
     ["A real file is written and decoded back bit-exactly (target 100%).",
      "CDFs are built on the CPU in float32 so encoder and decoder agree."]),
    ("Baselines and size prediction", "grey",
     ["JPEG, WebP, PCA transform codec, k-means vector quantisation, plain AE, compared by BD-rate on Kodak.",
      "Ridge / Lasso / Elastic Net predict file size from image features."]),
]
for i, (t, th, body) in enumerate(cards):
    col, row = i % 3, i // 3
    card(s, 0.6 + col * 4.1, 3.0 + row * 2.0, 3.95, 1.88, th, t, body, tsize=14.5, bsize=12.5)
notes(s, "Key idea: the entropy model is trained by maximum likelihood, so the bit cost is literally a negative log-likelihood. "
         "Rate-distortion: lambda trades bits against error, and six lambdas give six modes. Concepts map to courses 5 (GenAI), 4 (NN), 3 (Advanced ML), 2 and 1.")

# ───────────────────────────── Slide 5: concepts 2 ─────────────────────────────
s = prs.slides.add_slide(BLANK)
header(s, 5, "Concepts (2 of 2): realistic, faithful and honest",
       "Diffusion adds texture; guidance and protection keep it true; ML models decide the mode and warn about risk.")

cols = [
    ("GENERATE", "teal", [
        ("Diffusion, forward and backward", "Forward: add noise, q(xₜ|x₀), cosine schedule. Backward: an ε-prediction U-Net, trained as DDPM, sampled with DDIM in 25–50 steps."),
        ("Conditioning", "The U-Net also sees the coarse decode (channel concatenation), so it only has to add detail."),
        ("Classifier-free guidance", "10% of training drops the condition; at decode time scale w trades realism against fidelity.")]),
    ("PROTECT", "amber", [
        ("ROI protection", "Text (EasyOCR) and faces (MTCNN) become 16 px-aligned boxes, coded at the highest quality (λ = 0.0483) inside the file."),
        ("Bin-consistency guidance", "L_bin = mean(ReLU(|g_a(x) − ŷ| − 0.5 + m)²): at every denoising step, push x until it re-encodes to the transmitted integers."),
        ("ROI-lock guidance", "L_roi keeps protected boxes equal to their decoded crops, then pastes them with a feathered mask.")]),
    ("DECIDE", "rose", [
        ("Hallucination map", "Per 32×32 patch: seed variance, bin violation, detail gain. Isolation Forest, logistic regression and MLP score the risk; safe mode swaps in the coarse decode."),
        ("Rate control and recommender", "Ridge / XGBoost predict size and LPIPS per mode; kNN and a decision tree pick a mode for 2G / 3G / 4G / WiFi."),
        ("Failure mining", "Apriori rules such as {small text, low light, tiny mode} → hallucination become hard product rules.")]),
]
for i, (head, th, items) in enumerate(cols):
    x = 0.6 + i * 4.1
    soft, acc = THEMES[th]
    h = box(s, x, 1.6, 3.95, 0.4, fill=acc, line=None, radius=0.2)
    text(s, 0, 0, 0, 0, [[(head, {"size": 13, "bold": True, "color": WHITE})]], shape=h, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    for j, (t, d) in enumerate(items):
        y = 2.08 + j * 1.34
        box(s, x, y, 3.95, 1.26, fill=WHITE, line=LINE)
        box(s, x, y + 0.08, 0.07, 1.1, fill=acc, line=None, radius=0.5)
        text(s, x + 0.16, y + 0.03, 3.72, 1.22, [[(t, {"size": 13.5, "bold": True, "color": acc})], [(d, {"size": 11, "color": INK})]], space=2)

b = box(s, 0.6, 6.15, 12.13, 0.82, fill=THEMES["blue"][0], line=None)
text(s, 0.8, 6.17, 11.8, 0.8, [
    [("How we will judge it (guide targets, not yet results):  ", {"size": 11, "bold": True, "color": THEMES["blue"][1]}),
     ("BD-rate ≥ 30% vs JPEG · LPIPS ≥ 25% lower than the VAE alone · lower FID · Text Faithfulness Rate ≥ 2× plain diffusion (OCR error ≤ 0.05) · "
      "Face Identity Rate ≥ 1.5× (cosine ≥ 0.7) · hallucination AUROC ≥ 0.75 · decode bit-exact 100%.", {"size": 11})]], anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Left to right: generate, protect, decide. Bin-consistency guidance is the main contribution; it uses the transmitted bits as a hard constraint on the diffusion sampler. "
         "The bottom bar lists the success targets from the guide; training and full evaluation are still running, so do not present them as achieved results.")

prs.save(OUT)
print("saved", OUT)
