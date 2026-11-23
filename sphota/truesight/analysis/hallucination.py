"""Honesty layer, part 1 (Step 13): per-patch warning signals, labels, uncertainty overlay, safe mode, encoder-side boxes.

The decoded image is split into 32 x 32 patches. For each patch six numbers (the five signals of the guide; the coarse
texture signal has two parts) say how likely it is that the decoder invented the content:

    seed_var       decode with 3 seeds; per-pixel std, averaged per patch      bits did not decide the content
    bin_violation  |g_a(x_hat) - y_hat| beyond 0.5, upsampled x16, per patch   output inconsistent with the transmitted latent
    detail_gain    Laplacian variance of x_hat minus that of the coarse x_c    sharp detail on top of a blurry base
    coarse_lapvar  Laplacian variance of x_c                                   busy textures are where generators invent
    coarse_edge    edge density of x_c                                         (same signal, second measure)
    text_face      overlap with detected text / faces that are NOT protected   high-stakes content left unprotected

Label ("hallucinated" = sharp but wrong, the dangerous case): LPIPS(original patch, decoded patch) > 0.35 and the detail
gain is positive. The models that turn signals into a risk score live in risk_model.py.
Lapvar-type numbers are on the [0, 255] scale, like the Step 5 features. Images whose size is not a multiple of 32 are
cropped to the patch grid for analysis (the leftover border is not scored).
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..codec import bitstream as bs

PATCH = 32
LABEL_LPIPS = 0.35
N_SEEDS = 3
FEATURES = ["seed_var", "bin_violation", "detail_gain", "coarse_lapvar", "coarse_edge", "text_face"]
ENCODER_FEATURES = ["coarse_lapvar", "coarse_edge", "text_face"]     # computable by the encoder: coarse decode + detectors only
LOG_COLUMNS = ("detail_gain", "coarse_lapvar")
EDGE_THRESHOLD = 0.15                                                 # Sobel magnitude (gray in [0, 1]) that counts as an edge

_LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
_SOBEL_Y = _SOBEL_X.transpose(2, 3)


# ----------------------------------------------------------------------------------------------- patch helpers
def patch_grid(H, W, patch=PATCH):
    """Number of whole patches (rows, columns)."""
    return H // patch, W // patch


def to_patches(m, patch=PATCH):
    """Mean over every patch of a (1, 1, H, W) map -> (ph, pw) tensor. The border that does not fill a patch is dropped."""
    ph, pw = patch_grid(m.shape[-2], m.shape[-1], patch)
    return F.avg_pool2d(m[..., : ph * patch, : pw * patch], patch)[0, 0]


def gray255(x01):
    """(B, 3, H, W) in [0, 1] -> (B, 1, H, W) luma on the [0, 255] scale."""
    return (0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]) * 255.0


def laplacian_variance_map(x01, patch=PATCH):
    """Variance of the Laplacian response inside every patch -> (ph, pw)."""
    g = F.pad(gray255(x01), (1, 1, 1, 1), mode="replicate")
    lap = F.conv2d(g, _LAP.to(x01.device))
    mean, mean_sq = to_patches(lap, patch), to_patches(lap * lap, patch)
    return (mean_sq - mean ** 2).clamp(min=0)


def edge_density_map(x01, patch=PATCH, threshold=EDGE_THRESHOLD):
    """Fraction of pixels per patch whose Sobel gradient magnitude (gray in [0, 1]) exceeds ``threshold`` -> (ph, pw)."""
    g = F.pad(gray255(x01) / 255.0, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(g, _SOBEL_X.to(x01.device)), F.conv2d(g, _SOBEL_Y.to(x01.device))
    return to_patches((torch.sqrt(gx ** 2 + gy ** 2) > threshold).float(), patch)


def seed_variance_map(decodes, patch=PATCH):
    """Per-pixel std over the decodes of different seeds (population std, mean over colour channels), averaged per patch.
    ``decodes``: list of (1, 3, H, W) in [0, 1] (at least 2)."""
    if len(decodes) < 2:
        raise ValueError("seed variance needs at least 2 decodes")
    std = torch.stack([d.float() for d in decodes]).std(dim=0, unbiased=False)
    return to_patches(std.mean(dim=1, keepdim=True), patch)


@torch.no_grad()
def bin_violation_map(codec, x01, y_hat, patch=PATCH):
    """How far g_a(x_hat) leaves the transmitted rounding bins: relu(|g_a(x) - y_hat| - 0.5), mean over latent channels,
    upsampled x16 to pixels, averaged per patch -> (ph, pw). ``x01`` is edge-padded to the codec's multiple like the encoder did."""
    H, W = x01.shape[-2:]
    y = codec.g_a(bs.pad_to_multiple(x01.to(next(codec.parameters()).device), codec.DOWN))
    v = F.relu((y - y_hat.to(y.device)).abs() - 0.5).mean(dim=1, keepdim=True)
    up = F.interpolate(v, scale_factor=16, mode="nearest")[..., :H, :W]
    return to_patches(up.cpu(), patch)


def box_mask(H, W, boxes, xywh=False):
    """(1, 1, H, W) float mask, 1 inside any box. Boxes are (x0, y0, x1, y1), or (x, y, w, h) with xywh=True."""
    m = torch.zeros(1, 1, H, W)
    for b in boxes:
        x0, y0, x1, y1 = (b[0], b[1], b[0] + b[2], b[1] + b[3]) if xywh else b
        m[..., max(0, int(np.floor(y0))):min(H, int(np.ceil(y1))), max(0, int(np.floor(x0))):min(W, int(np.ceil(x1)))] = 1.0
    return m


def content_overlap_map(H, W, detected_boxes, protected_boxes, patch=PATCH):
    """Fraction of each patch covered by detected text / faces ((x0, y0, x1, y1) boxes) that is NOT inside a protected
    ((x, y, w, h)) box -> (ph, pw). Protected content is safe by construction, unprotected content is the risk."""
    unprotected = box_mask(H, W, detected_boxes) * (1 - box_mask(H, W, protected_boxes, xywh=True))
    return to_patches(unprotected, patch)


# ----------------------------------------------------------------------------------------------- LPIPS per patch / labels
_SPATIAL_LPIPS = None


def spatial_lpips_map(x, y):
    """AlexNet LPIPS distance map (1, 1, H, W) between two (1, 3, H, W) images in [0, 1] (its mean is the usual LPIPS)."""
    global _SPATIAL_LPIPS
    if _SPATIAL_LPIPS is None:
        import lpips
        _SPATIAL_LPIPS = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval()
    with torch.no_grad():
        return _SPATIAL_LPIPS(x.cpu() * 2 - 1, y.cpu() * 2 - 1)


def make_labels(lpips_patch, detail_gain, threshold=LABEL_LPIPS):
    """1 = hallucinated (sharp but wrong): patch LPIPS > ``threshold`` and positive detail gain. Arrays of equal shape."""
    return ((np.asarray(lpips_patch) > threshold) & (np.asarray(detail_gain) > 0)).astype(int)


# ----------------------------------------------------------------------------------------------- the feature table
def patch_table(decodes, coarse, y_hat, codec, detected_boxes=(), protected_boxes=(), original=None, lpips_fn=None,
                patch=PATCH):
    """All signals for one decoded image.

    decodes     list of >= 2 (1, 3, H, W) decodes of the same file with different seeds; decodes[0] is the shown image
    coarse      (1, 3, H, W) honest coarse decode;  y_hat the transmitted latent;  codec the low-lambda codec
    original    optional (1, 3, H, W): adds the per-patch LPIPS (``lpips_fn(original, decode)`` -> (1, 1, H, W) map;
                default spatial_lpips_map) needed for the labels
    Returns dict: X (n_patches, 6) in the FEATURES order, lpips (n_patches,) or None, grid (ph, pw), maps (name -> (ph, pw))."""
    x_hat = decodes[0].float().cpu()
    coarse = coarse.float().cpu()
    H, W = x_hat.shape[-2:]
    lap_hat, lap_coarse = laplacian_variance_map(x_hat, patch), laplacian_variance_map(coarse, patch)
    maps = {"seed_var": seed_variance_map(decodes, patch), "bin_violation": bin_violation_map(codec, x_hat, y_hat, patch),
            "detail_gain": lap_hat - lap_coarse, "coarse_lapvar": lap_coarse, "coarse_edge": edge_density_map(coarse, patch),
            "text_face": content_overlap_map(H, W, detected_boxes, protected_boxes, patch)}
    X = np.stack([maps[k].numpy().ravel() for k in FEATURES], axis=1).astype(np.float64)
    out = {"X": X, "lpips": None, "grid": tuple(maps["seed_var"].shape), "maps": {k: v.numpy() for k, v in maps.items()}}
    if original is not None:
        lp = (lpips_fn or spatial_lpips_map)(original.float().cpu(), x_hat)
        out["lpips"] = to_patches(lp.float(), patch).numpy().ravel().astype(np.float64)
    return out


def signed_log1p(v):
    return np.sign(v) * np.log1p(np.abs(v))


def transform_features(X, names=FEATURES):
    """Compress the heavy-tailed Laplacian-based columns with a signed log1p (the others are left as they are)."""
    X = np.array(X, dtype=np.float64, copy=True)
    for j, n in enumerate(names):
        if n in LOG_COLUMNS:
            X[:, j] = signed_log1p(X[:, j])
    return X


# ----------------------------------------------------------------------------------------------- what the user sees
def upsample_scores(scores, grid, H, W, patch=PATCH):
    """(n_patches,) or (ph, pw) scores -> smooth (1, 1, H, W) uncertainty map in [0, 1] (bilinear; the border outside the
    patch grid repeats the nearest patch)."""
    ph, pw = grid
    s = torch.as_tensor(np.asarray(scores), dtype=torch.float32).reshape(1, 1, ph, pw)
    up = F.interpolate(s, size=(ph * patch, pw * patch), mode="bilinear", align_corners=False)
    return F.pad(up, (0, W - pw * patch, 0, H - ph * patch), mode="replicate").clamp(0, 1)


def heat_overlay(x01, u, alpha=0.6):
    """Uncertainty heat map over the image: low risk stays clear, high risk turns red. x01 (1, 3, H, W), u (1, 1, H, W) in [0, 1]."""
    r = (1.5 - (4 * u - 3).abs()).clamp(0, 1)
    g = (1.5 - (4 * u - 2).abs()).clamp(0, 1)
    b = (1.5 - (4 * u - 1).abs()).clamp(0, 1)
    a = alpha * u
    return x01 * (1 - a) + torch.cat([r, g, b], dim=1) * a


def flag_pixel_mask(flags, H, W, patch=PATCH, feather=8):
    """(ph, pw) boolean patch flags -> soft (1, 1, H, W) pixel mask: flagged patches 1, others 0, edges blurred by a Gaussian
    of sigma = feather / 2 so no seams show."""
    ph, pw = flags.shape
    m = F.interpolate(torch.as_tensor(flags, dtype=torch.float32).view(1, 1, ph, pw), scale_factor=patch, mode="nearest")
    m = F.pad(m, (0, W - pw * patch, 0, H - ph * patch), mode="replicate")
    if feather > 0:
        sigma = feather / 2.0
        r = max(1, int(np.ceil(3 * sigma)))
        k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        m = F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return m.clamp(0, 1)


def safe_mode(x_hat, coarse, flags, protected_mask=None, patch=PATCH, feather=8):
    """Replace the decoder's content in flagged patches by the honest, blurry coarse decode (soft edges). Protected
    regions (hard mask (1, 1, H, W), the transmitted high-quality crops) are never replaced."""
    H, W = x_hat.shape[-2:]
    m = flag_pixel_mask(flags, H, W, patch, feather).to(x_hat.device)
    if protected_mask is not None:
        m = m * (1 - protected_mask.to(x_hat.device))
    return m * coarse.to(x_hat.device) + (1 - m) * x_hat


# ----------------------------------------------------------------------------------------------- the decoder-side pipeline
def analyse_decode(codec_lo, unet, sched, prot, risk_model, detected_boxes=(), seeds=range(N_SEEDS), steps=50, w=1.5,
                   threshold=0.5, guided=False, use_roi=True, original=None, patch=PATCH, feather=8, **guide_kw):
    """Decode the file with several seeds, score every patch and produce what the user sees.

    prot            roi.Protected bundle of the file (coarse decode, transmitted latent, protected canvas / mask / boxes)
    risk_model      fitted risk_model.RiskModel using the FULL signal set (features=FEATURES)
    detected_boxes  text / face boxes found in the decoded image, (x0, y0, x1, y1): those not protected count as risk
    Returns dict: x_hat (first-seed decode), decodes, table (patch_table output incl. X), risk (ph, pw) scores in [0, 1],
    flags (ph, pw) bool, overlay (heat map over x_hat), safe (x_hat with flagged patches replaced by the coarse decode)."""
    from ..diffusion.guidance import sphota_decode
    outs = [sphota_decode(codec_lo, None, unet, sched, prot=prot, w=w, steps=steps, seed=int(s), guided=guided,
                          use_roi=use_roi, **guide_kw) for s in seeds]
    decodes = [o.x for o in outs]
    tab = patch_table(decodes, prot.coarse, prot.y_hat, codec_lo, detected_boxes, prot.boxes if use_roi else (),
                      original=original, patch=patch)
    H, W = decodes[0].shape[-2:]
    risk = risk_model.predict_risk(tab["X"])
    grid = tab["grid"]
    flags = (risk >= threshold).reshape(grid)
    u = upsample_scores(risk, grid, H, W, patch)
    protected = prot.mask if (use_roi and len(prot.boxes)) else None
    return {"x_hat": decodes[0], "decodes": decodes, "table": tab, "risk": risk.reshape(grid), "flags": flags,
            "overlay": heat_overlay(decodes[0].cpu(), u.cpu()),
            "safe": safe_mode(decodes[0].cpu(), prot.coarse.cpu(), flags, None if protected is None else protected.cpu(),
                              patch, feather)}


# ----------------------------------------------------------------------------------------------- encoder-side protection
def risk_patch_boxes(flags, patch=PATCH):
    """(x0, y0, x1, y1) pixel boxes of the flagged patches."""
    ys, xs = np.nonzero(np.asarray(flags))
    return [(float(x * patch), float(y * patch), float((x + 1) * patch), float((y + 1) * patch)) for y, x in zip(ys, xs)]


def add_risk_boxes(base_raw_boxes, flags, H, W, patch=PATCH, max_boxes=16):
    """Protected boxes = detector boxes plus the high-risk patches, planned together (10% expansion, 16 px alignment,
    64 x 64 minimum, merging). The cap of ``max_boxes`` never drops a box that contains a detector box: surplus risk
    boxes (the smallest first) are dropped instead. Returns a list of (x, y, w, h)."""
    from ..protect import roi
    base = roi.plan_boxes(base_raw_boxes, H, W, max_boxes=10 ** 6)
    extra = roi.plan_boxes(risk_patch_boxes(flags, patch), H, W, max_boxes=10 ** 6)
    merged = roi.merge_boxes(base + extra)
    keep = lambda b: any(roi.overlaps(b, a) for a in base)
    while len(merged) > max_boxes:
        droppable = [b for b in merged if not keep(b)]
        if not droppable:
            break
        merged.remove(min(droppable, key=lambda b: b[2] * b[3]))
    return roi.select_boxes(merged, max_boxes)
