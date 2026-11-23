"""Rate control (Step 14a): predict the file size and quality of every mode from image features and pick a mode for a size target.

Mode table (long format): one row per (image x mode) with the Step 5 image features, the mode index 0..5 (the lambda index:
0 = smallest / lowest quality ... 5 = largest / highest quality), the measured ``bpp`` of that mode (real file size incl. header
and ROI streams, per original pixel) and ``lpips``. Columns ``image`` identify the image (used for image-level splits).

Models (all on log(bpp), the mode index one-hot encoded): Linear, Ridge, Lasso, Elastic Net, Random Forest, XGBoost; Random Forest and
XGBoost also predict LPIPS. Rate control for a target size S: take the HIGHEST-quality mode whose predicted size is <= S (the
smallest quality sacrifice that still meets the target; mode 0 if none does); if the measured size still exceeds S, step down
ONE mode (at most one retry).

Honest accuracy metrics (the six modes are discrete, so a target between two modes can only be approached from one side):
  pred_within    predicted size within +-10% of the measured size                      (the +-10% / 80% target of the guide)
  meets          measured size of the chosen mode <= target * 1.10 after <= 1 retry    (never ship a file that is too big)
  within_target  |measured - target| / target <= 10%  (strict; capped by what the six modes can reach: see oracle_within)
"""
import lpips  # noqa: F401  (must be imported BEFORE sklearn: the reverse order segfaults on some Windows installs)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import FEATURES

N_MODES = 6
SKEWED = ["grad_mean", "lap_var", "edge_density"]
TOLERANCE = 0.10
LINEAR_MODELS = ("linear", "ridge", "lasso", "elastic_net")
TREE_MODELS = ("random_forest", "xgboost")
SIZE_TARGET_FRACTION = 0.80


# ----------------------------------------------------------------------------------------------- design matrix
def design_matrix(df, n_modes=N_MODES):
    """Features -> (X, column names): log1p of the heavy-tailed Step 5 features, the 1/pixels term when the table has ``w`` and
    ``h`` (headers weigh more on small images: the Step 5 lesson), and the mode index one-hot encoded."""
    cols, names = [], []
    for f in FEATURES:
        v = df[f].to_numpy(dtype=float)
        cols.append(np.log1p(v) if f in SKEWED else v)
        names.append(f)
    if {"w", "h"} <= set(df.columns):
        # ONE resolution term: the fixed header cost adds ~1/pixels to bpp. A second one (log pixels) is collinear with it
        # whenever the table has only a few distinct image sizes and makes the unregularised linear model explode.
        px = (df["w"] * df["h"]).to_numpy(dtype=float)
        cols.append(1e5 / px)
        names.append("inv_pixels")
    m = df["mode"].to_numpy(dtype=int)
    if m.min() < 0 or m.max() >= n_modes:
        raise ValueError(f"mode index outside 0..{n_modes - 1}")
    for k in range(n_modes):
        cols.append((m == k).astype(float))
        names.append(f"mode_{k}")
    return np.column_stack(cols), names


def target_log_bpp(df):
    return np.log(df["bpp"].to_numpy(dtype=float))


# ----------------------------------------------------------------------------------------------- models
def make_models(groups=None, seed=0, n_splits=5):
    """Fresh, unfitted regressors. The linear ones choose their regularisation by cross-validation, grouped by image so
    that the modes of one image never sit in the training and the validation fold at once."""
    cv = None
    if groups is not None:
        g = np.asarray(groups)
        cv = list(GroupKFold(min(n_splits, len(np.unique(g)))).split(np.zeros(len(g)), groups=g))
    alphas = np.logspace(-4, 2, 25)
    return {
        # plain least squares; the one-hot mode columns are collinear with the intercept (equal, cancelling offsets, no effect)
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv)),
        "lasso": make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=cv or 5, max_iter=50000, random_state=seed)),
        "elastic_net": make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=alphas, cv=cv or 5,
                                                                    max_iter=50000, random_state=seed)),
        "random_forest": RandomForestRegressor(300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=seed, verbosity=0),
    }


def fit_models(train_df, target="log_bpp", names=None, seed=0):
    """Fit every model for ``target`` ('log_bpp' or 'lpips'). Returns {name: fitted model}."""
    X, _ = design_matrix(train_df)
    y = target_log_bpp(train_df) if target == "log_bpp" else train_df["lpips"].to_numpy(dtype=float)
    models = make_models(train_df["image"].to_numpy() if "image" in train_df else None, seed)
    models = {k: v for k, v in models.items() if names is None or k in names}
    return {k: m.fit(X, y) for k, m in models.items()}


def predict(model, df):
    return model.predict(design_matrix(df)[0])


def evaluate_models(models, test_df, target="log_bpp"):
    """R^2 of every fitted model on held-out rows, and for log_bpp also the share of rows whose predicted size is within +-10%."""
    X, _ = design_matrix(test_df)
    truth = target_log_bpp(test_df) if target == "log_bpp" else test_df["lpips"].to_numpy(dtype=float)
    out = {}
    for name, m in models.items():
        p = m.predict(X)
        row = {"r2": float(r2_score(truth, p))}
        if target == "log_bpp":
            row["within_10pct"] = float(np.mean(np.abs(np.exp(p) / np.exp(truth) - 1) <= TOLERANCE))
        out[name] = row
    return out


def lasso_features(lasso_model, names):
    """Features Lasso kept (non-zero coefficient) with their standardised coefficients, largest first."""
    coef = lasso_model[-1].coef_
    kept = {n: float(c) for n, c in zip(names, coef) if abs(c) > 1e-8}
    return dict(sorted(kept.items(), key=lambda kv: -abs(kv[1])))


# ----------------------------------------------------------------------------------------------- choosing a mode
def predicted_bytes(model, features, h, w, n_modes=N_MODES):
    """Predicted file size in bytes for every mode of ONE image. ``features``: mapping with the Step 5 FEATURES (and
    optionally w / h are taken from the arguments)."""
    row = {f: features[f] for f in FEATURES}
    df = pd.DataFrame([dict(row, mode=m, w=w, h=h) for m in range(n_modes)])
    return np.exp(model.predict(design_matrix(df, n_modes)[0])) * h * w / 8.0


def choose_mode(pred_bytes, target_bytes):
    """Highest-quality mode whose predicted size is <= target; mode 0 (the smallest file) if none fits."""
    ok = np.where(np.asarray(pred_bytes) <= target_bytes)[0]
    return int(ok.max()) if len(ok) else 0


def rate_control(model, features, h, w, target_bytes, measure=None, tolerance=TOLERANCE):
    """Pick a mode for ``target_bytes`` and, if ``measure(mode) -> measured bytes`` is given, verify it: when the measured
    size exceeds target * (1 + tolerance), step down ONE mode (at most one retry).
    Returns dict(mode, predicted_bytes, measured_bytes, retried, first_mode)."""
    pb = predicted_bytes(model, features, h, w)
    mode = first = choose_mode(pb, target_bytes)
    measured, retried = None, False
    if measure is not None:
        measured = measure(mode)
        if measured > target_bytes * (1 + tolerance) and mode > 0:
            mode, retried = mode - 1, True
            measured = measure(mode)
    return {"mode": mode, "first_mode": first, "predicted_bytes": float(pb[first]), "measured_bytes": measured, "retried": retried}


def make_targets(df, n_per_image=3, seed=0):
    """Size targets in bytes for every image of a mode table: log-uniform between the image's smallest and largest mode."""
    rng = np.random.default_rng(seed)
    rows = []
    for img, g in df.groupby("image"):
        b = (g["bpp"] * g["w"] * g["h"] / 8.0).to_numpy() if {"w", "h"} <= set(g.columns) else g["bytes"].to_numpy()
        lo, hi = np.log(b.min()), np.log(b.max())
        for t in np.exp(rng.uniform(lo, hi, n_per_image)):
            rows.append((img, float(t)))
    return pd.DataFrame(rows, columns=["image", "target_bytes"])


def evaluate_rate_control(model, test_df, targets, tolerance=TOLERANCE):
    """Run rate control on held-out images against the measured sizes of the table. Returns (per-target DataFrame, summary).

    The summary reports pred_within (size predictor accuracy), meets, within_target, the share of targets that needed the retry,
    and ``oracle_within`` / ``oracle_meets``: what a perfect predictor could reach with these six discrete modes."""
    rows = []
    for _, t in targets.iterrows():
        g = test_df[test_df.image == t.image].sort_values("mode")
        h, w = int(g.h.iloc[0]), int(g.w.iloc[0])
        feats = g.iloc[0]
        size = dict(zip(g["mode"], g["bpp"] * g["w"] * g["h"] / 8.0))
        r = rate_control(model, feats, h, w, t.target_bytes, measure=lambda m: size[m], tolerance=tolerance)
        rel = abs(r["measured_bytes"] - t.target_bytes) / t.target_bytes
        modes = np.array(sorted(size))
        best = min(abs(size[m] - t.target_bytes) / t.target_bytes for m in modes)
        fits = [m for m in modes if size[m] <= t.target_bytes * (1 + tolerance)]
        rows.append(dict(image=t.image, target_bytes=t.target_bytes, mode=r["mode"], retried=r["retried"],
                         measured_bytes=r["measured_bytes"], rel_error=rel, meets=r["measured_bytes"] <= t.target_bytes * (1 + tolerance),
                         within_target=rel <= tolerance, oracle_within=best <= tolerance, oracle_meets=bool(fits),
                         pred_rel_error=abs(r["predicted_bytes"] / size[r["first_mode"]] - 1)))
    d = pd.DataFrame(rows)
    summary = {"n_targets": int(len(d)), "images": int(d.image.nunique()), "meets": float(d.meets.mean()),
               "within_target": float(d.within_target.mean()), "retry_rate": float(d.retried.mean()),
               "pred_within": float((d.pred_rel_error <= tolerance).mean()), "oracle_within": float(d.oracle_within.mean()),
               "oracle_meets": float(d.oracle_meets.mean()), "mean_measured_over_target": float((d.measured_bytes / d.target_bytes).mean())}
    return d, summary


def size_prediction_summary(model, test_df, tolerance=TOLERANCE):
    """Share of held-out (image, mode) rows whose predicted size is within +-tolerance of the measured size, and per image
    (all six modes within tolerance) - the +-10% / 80% target of the guide."""
    p = np.exp(predict(model, test_df))
    ok = np.abs(p / test_df["bpp"].to_numpy() - 1) <= tolerance
    per_image = pd.Series(ok).groupby(test_df["image"].to_numpy()).mean()
    return {"rows_within": float(ok.mean()), "images_with_all_modes_within": float((per_image == 1.0).mean()),
            "images_majority_within": float((per_image >= 0.5).mean())}


def pca_embedding(df, n_components=2):
    """PCA of the standardised image features (one row per image) for the feature-space figure. Returns (coords, explained
    variance ratio, image ids, and the mean bpp of each image to colour the points)."""
    one = df.drop_duplicates("image")
    X = StandardScaler().fit_transform(design_matrix(one.assign(mode=0))[0][:, : len(FEATURES)])
    p = PCA(n_components, random_state=0)
    coords = p.fit_transform(X)
    bpp = df.groupby("image")["bpp"].mean().reindex(one["image"]).to_numpy()
    return coords, p.explained_variance_ratio_, one["image"].to_numpy(), bpp
