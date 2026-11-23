"""Honesty layer, part 2 (Step 13): the eight hallucination detectors, learning curves and the encoder-side risk model.

Input: the patch signals of hallucination.py (columns in the FEATURES order; ``seed_var`` first), label 1 = hallucinated.
All detectors turn the signals into a risk SCORE (higher = riskier) and are compared by AUROC on held-out IMAGES:
patches of one image are strongly correlated, so splitting by patch would leak - always use split_by_image.

    isolation_forest   unsupervised anomaly detection on the signals (no labels)                      Course 1
    gmm_seed_var       2-component GMM on the seed variance; posterior of the high-variance component  Course 5
    perceptron, mlp    neural baselines (standardised signals; MLP = 2 x 32 ReLU)                      Course 4
    logistic           logistic regression (class-balanced)                                            Course 2
    svm, random_forest, xgboost   kernel / tree models                                                 Course 3
"""
from pathlib import Path

import lpips  # noqa: F401  (must be imported BEFORE sklearn: the reverse order segfaults on some Windows installs)
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .hallucination import ENCODER_FEATURES, FEATURES, transform_features

TARGET_AUROC = 0.75
DETECTORS = ("isolation_forest", "gmm_seed_var", "perceptron", "logistic", "mlp", "svm", "random_forest", "xgboost")
SVM_MAX_TRAIN = 4000          # RBF-SVM with probabilities is O(n^2): fit on a class-stratified subsample of this size


def split_by_image(groups, test_frac=0.3, seed=0):
    """Train / test indices such that every image (group id) is entirely on one side. Splitting by patch would put
    near-identical neighbouring patches on both sides and inflate the AUROC."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(groups)), groups=np.asarray(groups)))
    return tr, te


def _auroc(y, s):
    if len(np.unique(y)) < 2:
        raise ValueError("AUROC needs both classes in the evaluation labels")
    return float(roc_auc_score(y, s))


def _subsample(X, y, n, seed=0):
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], max(1, int(n * (y == c).mean())), replace=False)
                          for c in np.unique(y)])
    return X[idx], y[idx]


def _supervised_models(ytr, seed=0):
    pos_weight = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    return {
        "perceptron": make_pipeline(StandardScaler(), Perceptron(random_state=seed, class_weight="balanced")),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=500, random_state=seed)),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(400, class_weight="balanced", random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight,
                                 random_state=seed, verbosity=0),
    }


def _score(name, model, X):
    return model.decision_function(X) if name == "perceptron" else model.predict_proba(X)[:, 1]


def evaluate_detectors(Xtr, ytr, Xte, yte, seed=0):
    """AUROC of all eight detectors, trained on (Xtr, ytr) and scored on (Xte, yte). The unsupervised ones do not use
    the labels. Returns {detector name: AUROC} in the DETECTORS order."""
    Xtr, Xte, ytr, yte = np.asarray(Xtr, float), np.asarray(Xte, float), np.asarray(ytr), np.asarray(yte)
    res = {}
    iso = IsolationForest(n_estimators=300, random_state=seed).fit(Xtr)
    res["isolation_forest"] = _auroc(yte, -iso.score_samples(Xte))
    gmm = GaussianMixture(2, random_state=seed).fit(Xtr[:, :1])                       # seed variance is column 0
    hi = int(np.argmax(gmm.means_[:, 0]))
    res["gmm_seed_var"] = _auroc(yte, gmm.predict_proba(Xte[:, :1])[:, hi])
    for name, m in _supervised_models(ytr, seed).items():
        Xf, yf = _subsample(Xtr, ytr, SVM_MAX_TRAIN, seed) if name == "svm" else (Xtr, ytr)
        m.fit(Xf, yf)
        res[name] = _auroc(yte, _score(name, m, Xte))
    return {k: res[k] for k in DETECTORS}


def best_detector(results):
    """(name, AUROC, met the 0.75 target) of the best detector in an AUROC dict."""
    name = max(results, key=results.get)
    return name, results[name], bool(results[name] >= TARGET_AUROC)


def learning_curve(model_name, Xtr, ytr, Xval, yval, fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0), seed=0):
    """Train and validation AUROC against the training-set size, for the bias-variance discussion: a big gap that closes
    with more data = variance, both low and close = bias. Returns a list of dicts (n_train, train_auroc, val_auroc);
    sizes whose subsample lacks one class are skipped."""
    Xtr, ytr, Xval, yval = np.asarray(Xtr, float), np.asarray(ytr), np.asarray(Xval, float), np.asarray(yval)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ytr))
    out = []
    for f in fractions:
        idx = order[: max(2, int(round(f * len(ytr))))]
        if len(np.unique(ytr[idx])) < 2:
            continue
        m = _supervised_models(ytr[idx], seed)[model_name]
        m.fit(Xtr[idx], ytr[idx])
        out.append({"n_train": int(len(idx)), "train_auroc": _auroc(ytr[idx], _score(model_name, m, Xtr[idx])),
                    "val_auroc": _auroc(yval, _score(model_name, m, Xval))})
    return out


class RiskModel:
    """Patch risk score from a logistic regression on a subset of the signals.

    ``features`` defaults to ENCODER_FEATURES (coarse texture + unprotected text/face overlap): those are computable on the
    ENCODER from the coarse decode alone, so the sender can add high-risk patches to the protected boxes before sending.
    Use features=FEATURES for the full decoder-side risk map. Inputs are raw signal tables (columns in FEATURES order)."""

    def __init__(self, features=tuple(ENCODER_FEATURES), seed=0):
        self.features, self.seed, self.model = tuple(features), seed, None
        self._cols = [FEATURES.index(f) for f in self.features]

    def _prep(self, X):
        return transform_features(np.asarray(X, float))[:, self._cols]

    def fit(self, X, y):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        self.model.fit(self._prep(X), np.asarray(y))
        return self

    def predict_risk(self, X):
        """Probability-like risk in [0, 1] per patch (row of X)."""
        if self.model is None:
            raise RuntimeError("RiskModel is not fitted")
        return self.model.predict_proba(self._prep(X))[:, 1]

    def flags(self, X, threshold=0.5):
        return self.predict_risk(X) >= threshold

    def save(self, path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": self.features, "seed": self.seed, "model": self.model}, path)

    @classmethod
    def load(cls, path):
        import joblib
        d = joblib.load(path)
        m = cls(d["features"], d["seed"])
        m.model = d["model"]
        return m
