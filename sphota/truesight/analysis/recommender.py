"""Mode recommender (Step 14b): "which mode for this photo on 2G?".

Modes (quality order, index 0..3):  tiny = lambda 0 + diffusion | small = lambda 1 + diffusion | balanced = lambda 3 + light
diffusion | exact = lambda 5, codec only (no generation). Context of a request: the image features (Step 5), the share of the
image covered by text and faces, and the network (2G / 3G / 4G / WiFi).

ORACLE (the label to learn): the cheapest mode that meets a quality bar (LPIPS <= 0.25) AND a faithfulness bar (no text or face
failure), where cost = transfer time on that network + decode time. If no mode meets both bars the best-quality mode (exact) is used.

Recommenders: kNN (k = 10 most similar past images, standardised features, Euclidean; the most frequent oracle mode of those
neighbours on the current network), a depth-4 decision tree with printable rules, a per-user history adjustment (a user who keeps
asking for the HD version is moved one mode up) and hard product rules mined from failure logs (rules.py). Evaluation: precision@1
against the oracle on held-out images.

Tables: ``context`` has one row per image (image, the 7 Step 5 features, text_frac, face_frac, text_h, n_faces);
``outcomes`` has one row per (image, mode name): image, mode, bytes, lpips, text_ok, face_ok [, decode_s] (NaN ok-flags mean
'no text / face in the image', which counts as ok).
"""
from dataclasses import dataclass, field

import lpips  # noqa: F401  (must be imported BEFORE sklearn: the reverse order segfaults on some Windows installs)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .features import FEATURES
from .rules import MODES, enforce_rules

NETWORKS = ("2G", "3G", "4G", "WiFi")
BANDWIDTH_KBPS = {"2G": 50.0, "3G": 1000.0, "4G": 12000.0, "WiFi": 50000.0}      # effective throughput
DECODE_SECONDS = {"tiny": 8.0, "small": 8.0, "balanced": 3.0, "exact": 0.3}        # diffusion decode dominates; measured on a GPU
MODE_SPECS = {"tiny": {"lambda_index": 0, "decoder": "diffusion"}, "small": {"lambda_index": 1, "decoder": "diffusion"},
              "balanced": {"lambda_index": 3, "decoder": "light"}, "exact": {"lambda_index": 5, "decoder": "codec"}}
QUALITY_BAR = 0.25
SKEWED = ["grad_mean", "lap_var", "edge_density"]
CONTEXT_COLS = FEATURES + ["text_frac", "face_frac"]
K = 10


# ----------------------------------------------------------------------------------------------- the oracle
def transfer_seconds(n_bytes, network):
    return n_bytes * 8.0 / (BANDWIDTH_KBPS[network] * 1000.0)


def _ok(v):
    return True if pd.isna(v) else bool(v)


def feasible(row, quality_bar=QUALITY_BAR):
    """Quality bar (LPIPS <= bar) and faithfulness bar (no text failure, no face failure) of one outcome row."""
    return bool(row["lpips"] <= quality_bar and _ok(row.get("text_ok", np.nan)) and _ok(row.get("face_ok", np.nan)))


def oracle_mode(outcomes, network, quality_bar=QUALITY_BAR, decode_seconds=None):
    """Index (into MODES) of the cheapest mode meeting both bars on ``network``; ``outcomes`` has one row per mode of ONE image.
    Cost = transfer time + decode time. No mode meets the bars -> the highest-quality one that exists in ``outcomes``."""
    dec = {**DECODE_SECONDS, **(decode_seconds or {})}
    best, best_cost = None, np.inf
    for _, r in outcomes.iterrows():
        if not feasible(r, quality_bar):
            continue
        cost = transfer_seconds(r["bytes"], network) + float(r["decode_s"] if "decode_s" in r and not pd.isna(r.get("decode_s")) else dec[r["mode"]])
        if cost < best_cost or (cost == best_cost and MODES.index(r["mode"]) > MODES.index(best)):
            best, best_cost = r["mode"], cost
    if best is None:
        best = max(outcomes["mode"], key=MODES.index)
    return MODES.index(best)


def build_oracle(outcomes, networks=NETWORKS, quality_bar=QUALITY_BAR, decode_seconds=None):
    """Oracle label for every (image, network): DataFrame image, network, oracle (mode index)."""
    rows = [{"image": img, "network": net, "oracle": oracle_mode(g, net, quality_bar, decode_seconds)}
            for img, g in outcomes.groupby("image") for net in networks]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- features
def context_matrix(context):
    """(n, 9) float matrix: log1p of the heavy-tailed features, the others unchanged, in CONTEXT_COLS order."""
    X = context[CONTEXT_COLS].to_numpy(dtype=float).copy()
    for j, c in enumerate(CONTEXT_COLS):
        if c in SKEWED:
            X[:, j] = np.log1p(X[:, j])
    return X


@dataclass
class ItemThresholds:
    """Boolean situations used in the failure log and the product rules."""
    small_text_px: float = 24.0        # text lines shorter than this (median height in px) are 'small text'
    many_faces: int = 3
    low_light: float = 60.0            # mean brightness on the 0-255 scale
    high_texture: float = float("inf")  # Laplacian-variance level; set from the training images by fit()

    @classmethod
    def fit(cls, context, quantile=0.75, **kw):
        return cls(high_texture=float(np.quantile(context["lap_var"], quantile)), **kw)


def item_flags(ctx, th):
    """Set of situation items true for one context row (mapping): small_text, many_faces, low_light, high_texture."""
    out = set()
    if ctx.get("text_frac", 0) > 0 and ctx.get("text_h", np.inf) < th.small_text_px:
        out.add("small_text")
    if ctx.get("n_faces", 0) >= th.many_faces:
        out.add("many_faces")
    if ctx["brightness"] < th.low_light:
        out.add("low_light")
    if ctx["lap_var"] >= th.high_texture:
        out.add("high_texture")
    return out


def failure_log(context, outcomes, th, halluc_col="halluc"):
    """Boolean log for rule mining, one row per (image, mode): situation items, mode_<name> items and HALLUC. HALLUC is true when
    the decode failed the faithfulness bar (text or face changed) or the ``halluc`` column of ``outcomes`` is true."""
    ctx = context.set_index("image")
    rows = []
    for _, o in outcomes.iterrows():
        c = ctx.loc[o["image"]]
        items = item_flags(c, th)
        row = {k: (k in items) for k in ("small_text", "many_faces", "low_light", "high_texture")}
        row.update({f"mode_{m}": (m == o["mode"]) for m in MODES})
        row["HALLUC"] = bool(not _ok(o.get("text_ok", np.nan)) or not _ok(o.get("face_ok", np.nan)) or
                             (halluc_col in o and not pd.isna(o[halluc_col]) and bool(o[halluc_col])))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- recommenders
class KNNRecommender:
    """k most similar past images (standardised features, Euclidean) -> most frequent oracle mode among them on this network.
    Ties go to the HIGHER-quality mode (the safer error)."""

    def __init__(self, k=K):
        self.k = k

    def fit(self, context, oracle):
        self.scaler = StandardScaler().fit(context_matrix(context))
        self.train_images = context["image"].to_numpy()
        self.Z = self.scaler.transform(context_matrix(context))
        self.labels = {net: g.set_index("image")["oracle"] for net, g in oracle.groupby("network")}
        return self

    def predict(self, context, network):
        """Oracle-mode indices for every row of ``context`` on ``network``."""
        Z = self.scaler.transform(context_matrix(context))
        lab = self.labels[network].reindex(self.train_images).to_numpy()
        out = []
        for z in Z:
            idx = np.argsort(((self.Z - z) ** 2).sum(1))[: self.k]
            counts = np.bincount(lab[idx].astype(int), minlength=len(MODES))
            out.append(int(np.max(np.where(counts == counts.max())[0])))
        return np.array(out)


class TreeRecommender:
    """Depth-4 decision tree on (image features, text / face share, network level 0=2G .. 3=WiFi) -> oracle mode."""

    def __init__(self, max_depth=4, min_samples_leaf=5, seed=0):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        self.names = CONTEXT_COLS + ["network"]

    def _X(self, context, networks):
        base = context_matrix(context)
        return np.column_stack([base, np.asarray([NETWORKS.index(n) for n in networks], dtype=float)])

    def fit(self, context, oracle):
        merged = oracle.merge(context, on="image")
        self.tree.fit(self._X(merged, merged["network"]), merged["oracle"])
        return self

    def predict(self, context, network):
        return self.tree.predict(self._X(context, [network] * len(context)))

    def export_rules(self):
        """sklearn export_text of the tree with readable feature names (features are log1p-scaled where heavy-tailed)."""
        head = "network: " + ", ".join(f"{i} = {n}" for i, n in enumerate(NETWORKS)) + "; mode: " + ", ".join(f"{i} = {m}" for i, m in enumerate(MODES))
        return head + "\n" + export_text(self.tree, feature_names=self.names)


class UserHistory:
    """Per-user count of 'asked for the HD (exact) version afterwards'. A user who does so in at least ``threshold`` of the
    last >= ``min_events`` requests is moved ``max_shift`` modes up."""

    def __init__(self, min_events=3, threshold=0.5, max_shift=1):
        self.min_events, self.threshold, self.max_shift = min_events, threshold, max_shift
        self.hd, self.total = {}, {}

    def record(self, user, requested_exact):
        self.hd[user] = self.hd.get(user, 0) + int(bool(requested_exact))
        self.total[user] = self.total.get(user, 0) + 1

    def hd_rate(self, user):
        return self.hd.get(user, 0) / self.total[user] if self.total.get(user) else 0.0

    def shift(self, user):
        return self.max_shift if (self.total.get(user, 0) >= self.min_events and self.hd_rate(user) >= self.threshold) else 0

    def adjust(self, mode_idx, user):
        return min(mode_idx + self.shift(user), len(MODES) - 1)


@dataclass
class Recommendation:
    mode: str
    index: int
    base_index: int
    reasons: list = field(default_factory=list)


class Recommender:
    """base model (kNN or tree) + per-user history + hard product rules. ``recommend`` explains every adjustment."""

    def __init__(self, model, history=None, product_rules=(), thresholds=None):
        self.model, self.history, self.rules, self.th = model, history, list(product_rules), thresholds or ItemThresholds()

    def recommend(self, ctx, network, user=None):
        """ctx: mapping with the CONTEXT_COLS (and optionally text_h, n_faces) of ONE image."""
        row = pd.DataFrame([{c: ctx[c] for c in CONTEXT_COLS}])
        base = int(self.model.predict(row, network)[0])
        idx, reasons = base, []
        if self.history is not None and user is not None:
            adj = self.history.adjust(idx, user)
            if adj != idx:
                reasons.append(f"user {user} often asks for the HD version: +{adj - idx} mode")
            idx = adj
        idx, applied = enforce_rules(idx, item_flags(ctx, self.th), self.rules)
        reasons += [f"rule: {a}" for a in applied]
        return Recommendation(MODES[idx], idx, base, reasons)


# ----------------------------------------------------------------------------------------------- evaluation
def precision_at_1(pred, oracle_idx):
    """Share of (image, network) pairs where the recommended mode equals the oracle mode."""
    return float(np.mean(np.asarray(pred) == np.asarray(oracle_idx)))


def evaluate_recommender(model, context_test, oracle_test, outcomes_test=None, quality_bar=QUALITY_BAR):
    """precision@1 overall and per network on held-out images. With ``outcomes_test`` also the share of recommendations that
    meet both quality bars ('adequate') and the mean ratio of the recommended cost to the oracle cost on that network."""
    ctx = context_test.set_index("image")
    res = {"per_network": {}}
    allp, allo, adequate, ratio = [], [], [], []
    out = outcomes_test.set_index("image") if outcomes_test is not None else None
    for net, g in oracle_test.groupby("network"):
        c = ctx.loc[g["image"]].reset_index()
        p = model.predict(c, net)
        o = g["oracle"].to_numpy()
        res["per_network"][net] = precision_at_1(p, o)
        allp.append(p); allo.append(o)
        if out is not None:
            for img, pi, oi in zip(g["image"], p, o):
                rows = out.loc[[img]].reset_index()
                by = rows.set_index("mode")
                adequate.append(feasible(by.loc[MODES[pi]], quality_bar) if MODES[pi] in by.index else False)
                cost = lambda m: transfer_seconds(by.loc[m, "bytes"], net) + DECODE_SECONDS[m]
                if MODES[pi] in by.index and MODES[oi] in by.index:
                    ratio.append(cost(MODES[pi]) / cost(MODES[oi]))
    res["precision_at_1"] = precision_at_1(np.concatenate(allp), np.concatenate(allo))
    if adequate:
        res["adequate"] = float(np.mean(adequate))
        res["mean_cost_ratio_vs_oracle"] = float(np.mean(ratio))
    return res


def simulate_users(base_modes, bytes_by_mode, n_users=200, sessions=30, hd_lover_share=0.3, p_lover=0.8, p_other=0.05,
                   history_kw=None, seed=0):
    """Simulated users for the report. Every user has a hidden probability of tapping 'HD' (request the exact version) after
    seeing a non-exact result: ``hd_lover_share`` of the users have ``p_lover``, the others ``p_other``. Each session draws one
    sample of (image, network) from ``base_modes`` (recommended mode index, shape (n,)) and ``bytes_by_mode`` (n, 4).
    Without adaptation the recommendation never changes; with UserHistory it moves HD lovers up one mode after a few taps.
    Returns the HD-request rate and the mean bytes per session for both, and the share of lovers that were shifted."""
    rng = np.random.default_rng(seed)
    base_modes, bytes_by_mode = np.asarray(base_modes), np.asarray(bytes_by_mode, dtype=float)
    last = len(MODES) - 1
    out = {"static": {"hd": 0, "n": 0, "bytes": 0.0}, "adaptive": {"hd": 0, "n": 0, "bytes": 0.0}}
    shifted_lovers, lovers = 0, 0
    for u in range(n_users):
        p = p_lover if rng.random() < hd_lover_share else p_other
        lovers += p == p_lover
        hist = UserHistory(**(history_kw or {}))
        for s in range(sessions):
            i = int(rng.integers(len(base_modes)))
            tap_draw = rng.random()
            for name, mode in (("static", int(base_modes[i])), ("adaptive", hist.adjust(int(base_modes[i]), u))):
                tapped = mode < last and tap_draw < p                              # same user behaviour in both arms
                out[name]["hd"] += int(tapped)
                out[name]["n"] += 1
                out[name]["bytes"] += bytes_by_mode[i, mode] + (bytes_by_mode[i, last] if tapped else 0.0)
                if name == "adaptive":
                    hist.record(u, tapped)
        shifted_lovers += (p == p_lover) and hist.shift(u) > 0
    res = {k: {"hd_request_rate": v["hd"] / v["n"], "mean_bytes": v["bytes"] / v["n"]} for k, v in out.items()}
    res["lovers_shifted"] = shifted_lovers / max(1, lovers)
    return res
