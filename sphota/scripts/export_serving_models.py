"""Export the trained intelligence artifacts for the website (experiments/serving/), from the Step 14 tables.

    python scripts/export_serving_models.py [--data experiments/step14_intelligence] [--out experiments/serving]

Reads mode_table.csv, context.csv, outcomes.csv (written by scripts/build_mode_table.py; they come back from the cloud run) and
fits the FINAL models on every image (train_intelligence.py only held images out to measure them):
    recommender.joblib   kNN + decision tree + product rules + thresholds + training context   -> /explain page and mode recommendation
    size_model.joblib    size and LPIPS predictors + residual std                               -> predicted vs actual on /explain
    failure_rules.json   mined {situation, mode} -> HALLUC rules                                -> rules list on /explain
risk_full.joblib (the calibrated hallucination risk model, Step 13) is not written here: copy it from the Step 13 output into
experiments/serving/ (with risk_meta.json) when you have it; until then /trust uses the measured seed-disagreement signal.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from truesight.analysis import rate_control as rc  # noqa: E402  (imports lpips before sklearn)
from truesight.analysis import recommender as rec  # noqa: E402
from truesight.analysis import rules as rl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "experiments" / "step14_intelligence"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "serving"))
    ap.add_argument("--min-support", type=float, default=0.03)
    ap.add_argument("--min-lift", type=float, default=1.2)
    ap.add_argument("--rule-confidence", type=float, default=0.5)
    ap.add_argument("--quality-bar", type=float, default=rec.QUALITY_BAR)
    ap.add_argument("--base", choices=("knn", "tree"), default="knn")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, context, outcomes = (pd.read_csv(data / f) for f in ("mode_table.csv", "context.csv", "outcomes.csv"))
    for c in ("text_ok", "face_ok"):
        if c not in outcomes:
            outcomes[c] = np.nan

    # ---- size / LPIPS predictors (the model type train_intelligence.py picked, else the best of the tree models)
    summ = data / "summary.json"
    best = json.loads(summ.read_text()).get("rate_control", {}).get("best_model") if summ.exists() else None
    m_size = rc.fit_models(table, "log_bpp")
    best = best if best in m_size else "random_forest"
    m_q = rc.fit_models(table, "lpips", names=rc.TREE_MODELS)
    lq = m_q.get(best) or m_q["random_forest"]
    X, _ = rc.design_matrix(table)
    resid = float(np.std(rc.target_log_bpp(table) - m_size[best].predict(X)))
    joblib.dump({"model": m_size[best], "lpips_model": lq, "resid_std": resid, "name": best}, out / "size_model.joblib")
    print(f"size_model.joblib  ({best}, residual std {resid:.3f} in log bpp)")

    # ---- recommender with product rules
    oracle = rec.build_oracle(outcomes, quality_bar=a.quality_bar)
    knn, tree = rec.KNNRecommender().fit(context, oracle), rec.TreeRecommender().fit(context, oracle)
    th = rec.ItemThresholds.fit(context)
    log = rec.failure_log(context, outcomes, th)
    rules = rl.mine_failure_rules(log, a.min_support, a.min_lift, top=15)
    product = rl.rules_to_product_rules(rules, a.rule_confidence)
    base = knn if a.base == "knn" else tree
    recommender = rec.Recommender(base, history=rec.UserHistory(), product_rules=product, thresholds=th)
    joblib.dump({"recommender": recommender, "tree": tree, "knn": knn, "thresholds": th, "context": context, "paths": None, "base": a.base},
                out / "recommender.joblib")
    print(f"recommender.joblib ({len(context)} photos, base {a.base}, {len(product)} product rules)")

    rows = [{"antecedents": sorted(map(str, r["antecedents"])), "support": float(r["support"]), "confidence": float(r["confidence"]),
             "lift": float(r["lift"])} for _, r in rules.iterrows()]
    (out / "failure_rules.json").write_text(json.dumps(rows, indent=2))
    print(f"failure_rules.json ({len(rows)} rules)\nRestart the API (python -m truesight.app.api) to load them.")


if __name__ == "__main__":
    main()
