import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from truesight.codec import bitstream as bs
from truesight.codec.entropy_gmm import torchac_available
from truesight.codec.vae_codec import TrueSightCodec
from truesight.protect import roi
from truesight.protect.detect import FaceDetector, TextDetector, filter_faces, text_boxes_from_easyocr
from truesight.protect.faithfulness import (EasyOCRAdapter, FaceEmbedder, face_identity_rate, text_faithfulness,
                                            text_faithfulness_rate)

ROOT = Path(__file__).resolve().parents[1]
needs_torchac = pytest.mark.skipif(not torchac_available(), reason="torchac cannot be imported (no C++ compiler on PATH)")


# --------------------------------------------------------------------------- box alignment
def _check_box(b, H, W, mult=16):
    x, y, w, h = b
    assert all(isinstance(v, int) for v in b)
    assert x % mult == 0 and y % mult == 0 and w % mult == 0 and h % mult == 0
    assert x >= 0 and y >= 0 and x + w <= W and y + h <= H and w > 0 and h > 0


def test_align_box_basic_properties():
    x, y, w, h = roi.align_box(100, 120, 180, 150, 512, 768)
    assert (x, y, w, h) == (96, 96, 96, 64) or (w % 16 == 0 and h % 16 == 0)       # 80 x 30 + 10% -> 96 x (min 64)
    _check_box((x, y, w, h), 512, 768)
    assert w >= 64 and h >= 64
    assert x <= 100 - 0.1 * 80 and x + w >= 180 + 0.1 * 80                          # covers the expanded box
    assert y <= 120 - 0.1 * 30 and y + h >= 150 + 0.1 * 30


def test_align_box_grows_tiny_boxes_to_the_minimum_size_and_keeps_them_inside():
    b = roi.align_box(5, 5, 15, 15, 400, 400)                                       # tiny box in the corner
    _check_box(b, 400, 400)
    assert b[:2] == (0, 0) and b[2:] == (64, 64)
    b = roi.align_box(395, 395, 399, 399, 400, 400)                                 # tiny box at the opposite corner
    _check_box(b, 400, 400)
    assert b[0] + b[2] == 400 // 16 * 16 and b[1] + b[3] == 400 // 16 * 16


def test_align_box_with_image_sizes_that_are_not_multiples_of_16():
    for H, W in [(500, 700), (333, 517), (100, 90), (70, 70), (47, 300)]:
        b = roi.align_box(10, 10, W - 5, H - 5, H, W)
        _check_box(b, H, W)
        assert b[2] <= W // 16 * 16 and b[3] <= H // 16 * 16                            # never into the unaligned border sliver


def test_align_box_returns_none_when_the_image_is_smaller_than_one_block():
    assert roi.align_box(0, 0, 5, 5, 10, 400) is None and roi.align_box(0, 0, 5, 5, 400, 15) is None


def test_align_box_random_boxes_always_satisfy_the_invariants():
    rng = random.Random(0)
    for _ in range(500):
        H, W = rng.randint(64, 900), rng.randint(64, 900)
        x0, y0 = rng.uniform(0, W - 2), rng.uniform(0, H - 2)
        x1, y1 = x0 + rng.uniform(1, W), y0 + rng.uniform(1, H)
        b = roi.align_box(x0, y0, min(x1, W), min(y1, H), H, W)
        _check_box(b, H, W)
        assert b[2] >= min(64, W // 16 * 16) and b[3] >= min(64, H // 16 * 16)


def test_align_box_covers_the_expanded_content_whenever_it_fits():
    rng = random.Random(1)
    for _ in range(300):
        H, W = 512, 768
        x0, y0 = rng.uniform(0, 600), rng.uniform(0, 400)
        x1, y1 = x0 + rng.uniform(5, 150), y0 + rng.uniform(5, 100)
        x, y, w, h = roi.align_box(x0, y0, x1, y1, H, W)
        bw, bh = x1 - x0, y1 - y0
        lo_x, hi_x = max(0.0, x0 - 0.1 * bw), min(float(W // 16 * 16), x1 + 0.1 * bw)     # the expanded box, clipped to the image
        lo_y, hi_y = max(0.0, y0 - 0.1 * bh), min(float(H // 16 * 16), y1 + 0.1 * bh)
        assert x <= lo_x + 1e-6 and x + w >= hi_x - 1e-6
        assert y <= lo_y + 1e-6 and y + h >= hi_y - 1e-6


# --------------------------------------------------------------------------- merging, capping, planning
def _no_overlaps(boxes):
    return not any(roi.overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def test_merge_overlapping_boxes_but_not_touching_ones():
    assert roi.merge_boxes([(0, 0, 64, 64), (32, 32, 64, 64)]) == [(0, 0, 96, 96)]
    assert sorted(roi.merge_boxes([(0, 0, 64, 64), (64, 0, 64, 64)])) == [(0, 0, 64, 64), (64, 0, 64, 64)]   # touching only
    chain = roi.merge_boxes([(0, 0, 64, 64), (48, 0, 64, 64), (96, 0, 64, 64), (300, 300, 64, 64)])
    assert sorted(chain) == [(0, 0, 160, 64), (300, 300, 64, 64)]                   # A-B and B-C overlap -> A,B,C merge
    assert roi.merge_boxes([]) == []


def test_merged_boxes_never_overlap_and_stay_aligned():
    rng = random.Random(2)
    for _ in range(100):
        raw = [(rng.randint(0, 40) * 16, rng.randint(0, 25) * 16, rng.randint(4, 12) * 16, rng.randint(4, 10) * 16)
               for _ in range(rng.randint(1, 14))]
        out = roi.merge_boxes(raw)
        assert _no_overlaps(out) and all(v % 16 == 0 for b in out for v in b)
        assert sum(b[2] * b[3] for b in out) >= max(b[2] * b[3] for b in raw)       # union never smaller than the biggest input


def test_select_boxes_keeps_the_largest_in_reading_order():
    boxes = [(i * 80, 0, 64 + 16 * (i % 3), 64) for i in range(20)]
    kept = roi.select_boxes(boxes, 16)
    assert len(kept) == 16 and kept == sorted(kept, key=lambda b: (b[1], b[0]))
    dropped = [b for b in boxes if b not in kept]
    assert max(b[2] * b[3] for b in dropped) <= min(b[2] * b[3] for b in kept)
    assert roi.select_boxes([(0, 0, 64, 64)], 16) == [(0, 0, 64, 64)]


class _Det:
    def __init__(self, boxes):
        self.boxes, self.seen = boxes, []

    def detect(self, img):
        self.seen.append(type(img).__name__)
        return self.boxes


def test_plan_and_find_protected_boxes_with_stand_in_detectors():
    text = _Det([(100, 100, 300, 140), (110, 130, 310, 170)])               # overlap after expansion -> merged
    faces = _Det([(500, 300, 580, 400)])
    img = Image.new("RGB", (768, 512))
    boxes, t, f = roi.find_protected_boxes(img, text, faces)
    assert len(boxes) == 2 and _no_overlaps(boxes) and t == text.boxes and f == faces.boxes
    for b in boxes:
        _check_box(b, 512, 768)
    # numpy input works too, either detector may be missing, and 40 scattered boxes are capped at 16
    assert roi.find_protected_boxes(np.zeros((512, 768, 3), np.uint8), None, faces)[0] == roi.plan_boxes(faces.boxes, 512, 768)
    many = _Det([(i % 8 * 90, i // 8 * 90, i % 8 * 90 + 30, i // 8 * 90 + 30) for i in range(40)])
    assert len(roi.plan_boxes(many.boxes, 800, 800)) <= 16
    assert roi.box_area_fraction([(0, 0, 256, 128)], 512, 512) == pytest.approx(0.125)


# --------------------------------------------------------------------------- detectors
def test_text_box_conversion_from_easyocr_output():
    horiz = [[10, 110, 20, 60], [5, 5, 5, 9]]                                # second one is degenerate (zero width)
    free = [[[0, 0], [30, 5], [28, 40], [2, 35]]]
    b = text_boxes_from_easyocr(horiz, free)
    assert b == [(10.0, 20.0, 110.0, 60.0), (0.0, 0.0, 30.0, 40.0)]


def test_face_filtering_by_probability_size_and_empty_results():
    boxes = np.array([[0, 0, 100, 100], [0, 0, 20, 100], [0, 0, 100, 100], [10, 10, 40, 40]], dtype=float)
    probs = np.array([0.99, 0.99, 0.85, 0.95])
    kept = filter_faces(boxes, probs)
    assert kept == [(0.0, 0.0, 100.0, 100.0), (10.0, 10.0, 40.0, 40.0)]       # 20 px wide and p=0.85 dropped; 30 px kept
    assert filter_faces(None, np.array([np.nan])) == [] and filter_faces(None, None) == []
    assert filter_faces(np.zeros((0, 4)), np.zeros(0)) == []
    assert filter_faces(np.array([[0, 0, 50, 50]], dtype=object), np.array([np.nan])) == []     # MTCNN's NaN probability


def test_detectors_are_lazy_and_swappable():
    td, fd = TextDetector(), FaceDetector()
    assert td._reader is None and fd._mtcnn is None                           # nothing heavy was loaded
    assert EasyOCRAdapter()._reader is None and FaceEmbedder()._model is None


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "kodak" / "kodim04.png").exists(), reason="Kodak not downloaded")
def test_real_mtcnn_finds_the_face_in_kodim04_and_none_in_a_landscape():
    pytest.importorskip("facenet_pytorch")
    det = FaceDetector()
    faces = det.detect(Image.open(ROOT / "data" / "raw" / "kodak" / "kodim04.png"))
    assert len(faces) == 1 and faces[0][0] < 200 < faces[0][2] and faces[0][1] < 400 < faces[0][3]
    assert det.detect(Image.open(ROOT / "data" / "raw" / "kodak" / "kodim01.png")) == []


@pytest.mark.skipif(not (Path.home() / ".EasyOCR" / "model" / "craft_mlt_25k.pth").exists(),
                    reason="EasyOCR CRAFT weights not on disk (would need a download)")
def test_real_easyocr_detector_finds_two_text_lines():
    pytest.importorskip("easyocr")
    cv2 = pytest.importorskip("cv2")
    img = np.full((384, 640, 3), 235, np.uint8)
    cv2.putText(img, "STOP 5 mg", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 5)
    cv2.putText(img, "Total 4,750", (60, 260), cv2.FONT_HERSHEY_DUPLEX, 1.8, (30, 30, 120), 4)
    boxes = TextDetector().detect(img)
    centres = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes]
    assert any(abs(cx - 200) < 120 and abs(cy - 100) < 50 for cx, cy in centres)      # first line
    assert any(abs(cx - 215) < 120 and abs(cy - 245) < 50 for cx, cy in centres)      # second line


# --------------------------------------------------------------------------- feathering and paste
def _box_mask(H, W, x, y, w, h):
    m = torch.zeros(1, 1, H, W)
    m[..., y:y + h, x:x + w] = 1
    return m


def test_feather_mask_is_one_inside_zero_outside_and_ramps_inside_the_box_edge():
    m = _box_mask(160, 160, 48, 48, 64, 64)
    s = roi.feather_mask(m, 8)
    assert s.shape == m.shape and (s <= m + 1e-6).all() and (s[m == 0] == 0).all()   # never leaks outside the box
    assert s[..., 80, 80].item() == pytest.approx(1.0, abs=1e-3)                    # deep inside
    row = s[0, 0, 80, 48:80]                                                        # left edge -> centre
    assert row[0] < 0.1 and (row.diff() >= -1e-6).all() and row[-1] > 0.99          # smooth monotone ramp 0 -> 1


def test_feather_mask_leaves_image_borders_alone_and_handles_zero_width():
    m = _box_mask(128, 128, 0, 0, 64, 64)                                           # box touching the top-left corner
    s = roi.feather_mask(m, 8)
    assert s[..., 0, 0].item() == pytest.approx(1.0, abs=1e-3) and s[..., 0, 30].item() > 0.99    # no fade at the border
    assert s[..., 30, 63].item() < 0.5                                              # but there is one at the inner edge
    assert torch.equal(roi.feather_mask(m, 0), m)
    assert torch.count_nonzero(roi.feather_mask(torch.zeros(1, 1, 32, 32))) == 0


def test_paste_blends_linearly():
    x, canvas = torch.zeros(1, 3, 4, 4), torch.ones(1, 3, 4, 4)
    for a in (0.0, 0.25, 1.0):
        assert torch.allclose(roi.paste(x, canvas, torch.full((1, 1, 4, 4), a)), torch.full((1, 3, 4, 4), a))


# --------------------------------------------------------------------------- ROI streams and file accounting
def test_roi_stream_roundtrip_and_truncation():
    z, y = b"abc", b"defgh"
    assert roi.unpack_roi_stream(roi.pack_roi_stream(z, y)) == (z, y)
    assert roi.unpack_roi_stream(roi.pack_roi_stream(b"", b"")) == (b"", b"")
    with pytest.raises(ValueError):
        roi.unpack_roi_stream(roi.ROI_STREAM.pack(99) + b"xy")


def test_size_breakdown_counts_every_roi_byte_in_the_total():
    boxes = [bs.Box(0, 0, 64, 64, b"x" * 100), bs.Box(64, 64, 64, 64, b"y" * 50)]
    blob = bs.pack(256, 256, 0, 1, 10, b"z" * 40, b"w" * 60, boxes)
    b = roi.size_breakdown(blob)
    assert b["total_bytes"] == len(blob) == bs.header_bytes(2) + 40 + 60 + 150
    assert b["roi_bytes"] == 2 * 12 + 150 and b["n_boxes"] == 2 and b["roi_stream_bytes"] == 150
    assert b["base_bytes"] + b["roi_bytes"] == b["total_bytes"]
    assert b["roi_share"] == pytest.approx(b["roi_bytes"] / b["total_bytes"])
    assert b["bpp_total"] == pytest.approx(b["bpp_base"] + b["bpp_roi"]) and b["bpp_total"] == pytest.approx(8 * len(blob) / 256 ** 2)
    assert roi.size_breakdown(bs.pack(64, 64, 0, 1, 10, b"", b""))["roi_share"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- full protected-region files (torchac)
def _net(seed):
    torch.manual_seed(seed)
    return TrueSightCodec(N=16, M=24, K=3).eval()


@needs_torchac
def test_protected_file_roundtrip_with_two_codecs():
    lo, hi = _net(0), _net(1)
    torch.manual_seed(3)
    x = torch.rand(1, 3, 128, 192)
    boxes = [(16, 32, 64, 48), (96, 64, 80, 64)]                              # 48 px high: padded to 64 for the hyperprior
    blob, d = roi.encode_protected(lo, hi, x, boxes, mode=2, seed=7)
    s = bs.unpack(blob)
    assert [(b.x, b.y, b.w, b.h) for b in s.boxes] == boxes and (s.mode, s.seed) == (2, 7)
    p = roi.decode_protected(_net(0), _net(1), blob)                           # fresh codec instances, like a receiver
    assert p.boxes == boxes and p.coarse.shape == x.shape and p.canvas.shape == x.shape and p.mask.shape == (1, 1, 128, 192)
    # base layer unchanged by the boxes: same latent as a file without ROI
    assert torch.equal(p.y_hat, d["y_hat"])
    # each box equals the high-codec decode of that crop (bit-exact latent), outside the boxes the canvas is empty
    for X, Y, w, h in boxes:
        crop = bs.pad_to_multiple(x[..., Y:Y + h, X:X + w], 64)
        ref = hi.g_s(torch.round(hi.g_a(crop))).clamp(0, 1)[..., :h, :w]
        assert torch.allclose(p.canvas[..., Y:Y + h, X:X + w], ref, atol=1e-5)
        assert (p.mask[..., Y:Y + h, X:X + w] == 1).all()
    assert p.mask.sum().item() == sum(w * h for _, _, w, h in boxes)
    assert torch.count_nonzero(p.canvas * (1 - p.mask)) == 0
    assert (p.soft_mask <= p.mask + 1e-6).all() and p.soft_mask.max() > 0.99
    # accounting: ROI bytes are inside the file and reported as a share of it
    br = roi.size_breakdown(blob)
    assert br["total_bytes"] == len(blob) and br["roi_bytes"] > 0 and 0 < br["roi_share"] < 1
    plain, _ = bs.encode_image(lo, x, mode=2, seed=7)
    assert len(blob) - len(plain) == br["roi_bytes"]                            # the boxes account for the whole difference


@needs_torchac
def test_file_without_boxes_decodes_to_an_empty_canvas():
    lo, hi = _net(0), _net(1)
    blob, _ = roi.encode_protected(lo, hi, torch.rand(1, 3, 64, 64), [])
    p = roi.decode_protected(lo, hi, blob)
    assert p.boxes == [] and p.mask.sum() == 0 and torch.count_nonzero(p.canvas) == 0 and torch.count_nonzero(p.soft_mask) == 0
    assert roi.size_breakdown(blob)["roi_bytes"] == 0


# --------------------------------------------------------------------------- Text Faithfulness Rate
class FakeOCR:
    """read(original) returns scripted regions; recognize(decoded, box) returns scripted text keyed by the decode's tag."""
    def __init__(self, regions, readings):
        self.regions, self.readings = regions, readings

    def read(self, img):
        return self.regions[img]

    def recognize(self, img, box):
        return self.readings[(img, box)]


def test_tfr_cer_threshold_is_inclusive_at_0_05():
    text = "ABCDEFGHIJKLMNOPQRST"                                              # 20 characters
    box = (0, 0, 10, 10)
    ocr = FakeOCR({"o": [(box, text, 0.9)]}, {("one_err", box): "ABCDEFGHIJKLMNOPQRSX", ("two_err", box): "ABCDEFGHIJKLMNOPQRXX",
                                               ("same", box): text})
    assert text_faithfulness(ocr, "o", "same")["passed"] and text_faithfulness(ocr, "o", "one_err")["passed"]   # CER = 0.05
    r = text_faithfulness(ocr, "o", "two_err")
    assert not r["passed"] and r["cers"][0] == pytest.approx(0.10)


def test_tfr_needs_all_confident_regions_to_pass_and_ignores_the_rest():
    b1, b2, b3 = (0, 0, 10, 10), (0, 20, 10, 30), (0, 40, 10, 50)
    ocr = FakeOCR({"o": [(b1, "STOP", 0.9), (b2, "5 mg", 0.8), (b3, "blurry", 0.3)], "none": [], "low": [(b1, "x", 0.2)]},
                  {("d", b1): "STOP", ("d", b2): "8 mg", ("d", b3): "zzzzzz", ("e", b1): "STOP", ("e", b2): "5 mg", ("e", b3): "qqq"})
    r = text_faithfulness(ocr, "o", "d")
    assert r["n_regions"] == 2 and r["n_ok"] == 1 and not r["passed"]          # "5 mg" -> "8 mg" fails; conf 0.3 region ignored
    assert text_faithfulness(ocr, "o", "e")["passed"]                            # the low-confidence region may differ freely
    assert text_faithfulness(ocr, "none", "d") is None and text_faithfulness(ocr, "low", "d") is None


def test_tfr_reports_each_region_with_both_readings():
    b1, b2 = (0, 0, 10, 10), (0, 20, 10, 30)
    ocr = FakeOCR({"o": [(b1, "STOP", 0.9), (b2, "5 mg", 0.8)]}, {("d", b1): "STOP", ("d", b2): "8 mg"})
    r = text_faithfulness(ocr, "o", "d")
    assert [g["original"] for g in r["regions"]] == ["STOP", "5 mg"] and [g["decoded"] for g in r["regions"]] == ["STOP", "8 mg"]
    assert r["regions"][1]["box"] == (0.0, 20.0, 10.0, 30.0) and r["regions"][1]["cer"] == pytest.approx(0.25)


def test_tfr_over_a_set_counts_only_eligible_images():
    b = (0, 0, 10, 10)
    ocr = FakeOCR({"a": [(b, "HELLO", 0.9)], "b": [(b, "WORLD", 0.9)], "c": [(b, "ABCD", 0.9)], "empty": []},
                  {("A", b): "HELLO", ("B", b): "WORXD", ("C", b): "ABCD", ("E", b): ""})
    r = text_faithfulness_rate(ocr, [("a", "A"), ("b", "B"), ("c", "C"), ("empty", "E")])
    assert r["n_images"] == 4 and r["n_eligible"] == 3 and r["n_pass"] == 2 and r["tfr"] == pytest.approx(200 / 3)
    assert r["region_rate"] == pytest.approx(200 / 3)
    assert np.isnan(text_faithfulness_rate(ocr, [("empty", "E")])["tfr"])


# --------------------------------------------------------------------------- Face Identity Rate
class ColourEmbedder:
    """Stand-in embedder: the mean colour of the crop is the 'identity'."""
    def embed(self, pil):
        return np.asarray(pil.convert("RGB"), dtype=float).reshape(-1, 3).mean(0)


def _img(colour):
    return Image.new("RGB", (200, 200), colour)


def test_fir_passes_at_cosine_0_7_and_counts_faces():
    det = _Det([(40, 40, 120, 120)])
    emb = ColourEmbedder()
    orig = _img((255, 0, 0))
    ok = face_identity_rate(det, emb, [(orig, _img((200, 150, 0)))])              # cosine 0.8
    bad = face_identity_rate(det, emb, [(orig, _img((150, 200, 0)))])             # cosine 0.6
    assert ok["fir"] == 100.0 and ok["n_faces"] == 1 and ok["per_image"][0][0]["cosine"] == pytest.approx(0.8)
    assert bad["fir"] == 0.0 and bad["per_image"][0][0]["cosine"] == pytest.approx(0.6)


def test_fir_mixed_set_and_images_without_faces():
    class PerImage:
        def detect(self, img):
            return [(10, 10, 100, 100)] if img.getpixel((0, 0))[0] > 100 else []
    pairs = [(_img((255, 0, 0)), _img((255, 10, 0))), (_img((255, 0, 0)), _img((0, 255, 0))), (_img((0, 0, 255)), _img((0, 0, 255)))]
    r = face_identity_rate(PerImage(), ColourEmbedder(), pairs)
    assert r["n_faces"] == 2 and r["n_pass"] == 1 and r["fir"] == 50.0 and r["n_images_with_faces"] == 2 and r["n_images"] == 3
    assert np.isnan(face_identity_rate(PerImage(), ColourEmbedder(), [pairs[2]])["fir"])


def test_fir_uses_the_same_box_for_original_and_decode():
    seen = []

    class Rec:
        def embed(self, pil):
            seen.append(pil.size)
            return np.ones(3)
    face_identity_rate(_Det([(40, 40, 120, 120)]), Rec(), [(_img((1, 2, 3)), _img((4, 5, 6)))])
    assert len(seen) == 2 and seen[0] == seen[1]
