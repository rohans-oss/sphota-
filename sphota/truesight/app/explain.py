"""The "why this mode" views of the web app, built from the REAL intelligence-layer models (Step 14): the decision-tree path of the
mode recommender, the failure-mining rules that fired, the nearest past photos (kNN) and the size / LPIPS prediction of the rate-control
model. Pure numpy / pandas / scikit-learn: no torch, no images."""
import numpy as np
import pandas as pd

from .config import MODES, NETWORKS

SKEWED = ("grad_mean", "lap_var", "edge_density")           # log1p-scaled inside the models (recommender.SKEWED)
PRETTY = {"text_frac": "text area", "face_frac": "face area", "grad_mean": "gradient", "lap_var": "sharpness",
          "entropy": "entropy", "edge_density": "edge density", "colorfulness": "colour", "brightness": "brightness",
          "jpeg50_bpp": "JPEG-50 bpp", "network": "network"}


def mode_title(m):
    return str(m)[:1].upper() + str(m)[1:]


def _raw(name, v):
    """Model-space value -> the unit a person reads (log1p-scaled features are inverted)."""
    return float(np.expm1(v)) if name in SKEWED else float(v)


def _fmt(name, v):
    if name in ("text_frac", "face_frac"):
        return f"{100 * v:.1f}%"
    if name == "network":
        return NETWORKS[int(min(max(round(v), 0), len(NETWORKS) - 1))]
    return f"{v:.3g}"


def split_label(name, threshold):
    """'text area ≤ 4.0%' for one split of the tree (``threshold`` is in the model's space)."""
    if name == "network":
        return f"network ≤ {NETWORKS[int(min(max(np.floor(threshold), 0), len(NETWORKS) - 1))]}"
    return f"{PRETTY.get(name, name)} ≤ {_fmt(name, _raw(name, threshold))}"


def image_value(name, ctx, network):
    """The value of this photo for a split feature, as text (the edge label on the taken path)."""
    if name == "network":
        return network
    return _fmt(name, float(ctx[name]))


def tree_view(tree_rec, ctx, network):
    """The decision tree of the recommender as the Explain page draws it: the path this photo takes is expanded and every
    node on it shows both children (the sibling off the path as a grey node), exactly like a path explanation.
    Returns ({"nodes": [...], "edges": [...]}, index of the mode the tree predicts)."""
    clf = tree_rec.tree
    t = clf.tree_
    x = tree_rec._X(pd.DataFrame([{k: ctx[k] for k in tree_rec.names if k != "network"}]), [network])
    path = [int(n) for n in clf.decision_path(x.astype(np.float32)).indices]
    on_path = set(path)
    leaf_mode = lambda n: int(clf.classes_[int(np.argmax(t.value[n][0]))])
    is_leaf = lambda n: t.children_left[n] == t.children_right[n]       # both -1

    shown = []                                                         # displayed nodes, in left-to-right (in-order) order
    parent_of, branch_of = {}, {}

    def walk(n, depth):
        if n in on_path and not is_leaf(n):
            left, right = int(t.children_left[n]), int(t.children_right[n])
            for c, b in ((left, "yes"), (right, "no")):
                parent_of[c], branch_of[c] = n, b
            walk(left, depth + 1)
            shown.append((n, depth))
            walk(right, depth + 1)
        else:
            shown.append((n, depth))

    walk(0, 0)
    slots = {n: i for i, (n, _) in enumerate(shown)}
    leaves = [n for n, _ in shown if n not in on_path or is_leaf(n)]
    n_leaf = max(len(leaves), 1)
    col = {}
    for i, n in enumerate(leaves):
        col[n] = (i + 0.5) / n_leaf
    for n, _ in reversed(shown):                                        # parents sit above the middle of their children
        if n not in col:
            col[n] = float(np.mean([col[c] for c in (int(t.children_left[n]), int(t.children_right[n])) if c in col]))
    final = path[-1]
    nodes, edges = [], []
    for n, depth in shown:
        leaf = bool(is_leaf(n))
        label = f"→ {mode_title(MODES[leaf_mode(n)])}" if leaf else split_label(tree_rec.names[int(t.feature[n])], t.threshold[n])
        nodes.append({"id": f"n{n}", "label": label, "depth": depth, "col": round(col[n], 3), "taken": n in on_path,
                      "leaf": leaf, "chosen": bool(n == final)})
    for c, p in parent_of.items():
        taken = c in on_path
        label = branch_of[c]
        if taken:
            label += " · " + image_value(tree_rec.names[int(t.feature[p])], ctx, network)
        edges.append({"from": f"n{p}", "to": f"n{c}", "label": label, "taken": taken})
    return {"nodes": nodes, "edges": edges}, leaf_mode(final)


def rules_view(rules, items, base_mode, final_mode, reasons, limit=5):
    """The failure-mining rules ({situation, mode} -> HALLUC) for this photo. A rule 'fired' when every item of its antecedent is true
    for the photo (situation items) and its mode item is the base recommendation. ``reasons`` are the strings of
    Recommender.recommend; a rule that raised the mode gets a note."""
    have = set(items) | {f"mode_{base_mode}"}
    out = []
    for r in rules:
        ante = list(r["antecedents"])
        fired = set(ante) <= have
        row = {"antecedents": ante, "consequent": "HALLUC", "support": float(r["support"]), "confidence": float(r["confidence"]),
               "lift": float(r["lift"]), "fired": bool(fired)}
        if fired:
            situation = " AND ".join(sorted(a for a in ante if not a.startswith("mode_")))
            if any(s.startswith(f"rule: {situation} =>") for s in reasons):
                row["note"] = f"Changed the mode from {mode_title(base_mode)} to {mode_title(final_mode)}"
        out.append(row)
    out.sort(key=lambda r: (not r["fired"], -r["lift"], -r["confidence"]))
    return out[:limit]


def neighbours_view(knn, ctx, network, k=4):
    """The k most similar past photos (the kNN of the recommender) and the mode that was best for each on this network.
    Returns [(image id in the training table, best mode name)]."""
    from ..analysis.recommender import context_matrix
    z = knn.scaler.transform(context_matrix(pd.DataFrame([ctx])))[0]
    order = np.argsort(((knn.Z - z) ** 2).sum(1))[:k]
    images = knn.train_images[order]
    lab = knn.labels[network].reindex(images).to_numpy()
    return [(images[i], MODES[int(lab[i])]) for i in range(len(images)) if not np.isnan(lab[i])]


def size_view(size_art, feats, h, w, lam):
    """Predicted size (KB, ± one residual std) and LPIPS of the codec decode for lambda index ``lam``, from the rate-control models.
    ``size_art`` = {"model", "lpips_model", "resid_std"} as written by scripts/export_serving_models.py."""
    from ..analysis import rate_control as rc
    pb = rc.predicted_bytes(size_art["model"], feats, h, w)
    kb = float(pb[lam]) / 1000.0
    std = kb * (float(np.exp(size_art.get("resid_std", 0.0))) - 1.0)
    lp = None
    if size_art.get("lpips_model") is not None:
        row = pd.DataFrame([dict({f: feats[f] for f in rc.FEATURES}, mode=lam, w=w, h=h)])
        lp = float(size_art["lpips_model"].predict(rc.design_matrix(row)[0])[0])
    return {"kb": kb, "kb_std": std, "lpips": lp}
