"""Loading of the trained intelligence / honesty artifacts that the web service needs besides the checkpoints.

They are written by scripts/export_serving_models.py from the outputs of the Kaggle run (experiments/step14_intelligence,
experiments/step13_honesty) into experiments/serving/. Each one is optional: when a file is absent the matching feature of the
site reports that clearly instead of inventing a value."""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Artifacts:
    recommender: object = None        # analysis.recommender.Recommender (base model + user history + product rules)
    tree: object = None               # TreeRecommender, for the decision-path view
    knn: object = None                # KNNRecommender, for the nearest-photos view
    thresholds: object = None         # ItemThresholds
    context: object = None            # DataFrame of the training photos (image, name, features ...)
    paths: list = None                # image file of training photo n (or None when the list was not available)
    base: str = "knn"
    size: dict = None                 # {"model", "lpips_model", "resid_std", "name"}
    risk: object = None               # analysis.risk_model.RiskModel with the FULL signal set
    risk_meta: dict = field(default_factory=lambda: {"guided": False, "with_roi": False, "threshold": 0.5})
    rules: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def load_artifacts(settings):
    """Load whatever exists in settings.serving_dir. Never raises for an absent file: it is listed in ``missing``."""
    import joblib
    from ..analysis import recommender as _rec  # noqa: F401  (imports lpips BEFORE sklearn, as the test conftest requires)
    from ..analysis.risk_model import RiskModel
    files = settings.optional_files()
    a = Artifacts()
    if files["recommender"].exists():
        d = joblib.load(files["recommender"])
        a.recommender, a.tree, a.knn = d["recommender"], d["tree"], d["knn"]
        a.thresholds, a.context, a.paths, a.base = d["thresholds"], d["context"], d.get("paths"), d.get("base", "knn")
    else:
        a.missing.append("recommender")
    if files["size_model"].exists():
        a.size = joblib.load(files["size_model"])
    else:
        a.missing.append("size_model")
    if files["risk_model"].exists():
        a.risk = RiskModel.load(files["risk_model"])
        meta = files["risk_model"].with_name("risk_meta.json")
        if meta.exists():
            a.risk_meta.update(json.loads(meta.read_text()))
    else:
        a.missing.append("risk_model")
    if files["failure_rules"].exists():
        a.rules = json.loads(files["failure_rules"].read_text())
    else:
        a.missing.append("failure_rules")
    return a
