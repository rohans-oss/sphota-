"""Step 14: train and evaluate rate control, the mode recommender and the failure-mining rules.

    python scripts/build_mode_table.py ...                      # first: builds mode_table.csv, context.csv, outcomes.csv
    python scripts/train_intelligence.py --data experiments/step14_intelligence

14a Rate control   Linear / Ridge / Lasso / Elastic Net / Random Forest / XGBoost predict log(bpp) (+ LPIPS for the tree models);
                   R^2, Lasso's chosen features, the +-10% size-prediction rate (target: >= 80%), and rate control for random size
                   targets on held-out images (meets / within / retry rate, with the ceiling the six discrete modes allow).
14b Recommender    oracle = cheapest mode meeting LPIPS <= 0.25 and no text / face failure (+ transfer and decode time per network);
                   kNN (k = 10) and a depth-4 decision tree (rules printed), precision@1 on held-out images (target: >= 0.6),
                   and a simulated-user study of the per-user history adjustment.
14c Failure rules  Apriori on the failure log (situation items + mode + HALLUC), the top rules with support / confidence / lift,
                   the product rules derived from them and what they do to the recommendations.
All splits are by image. Writes summary.json, tables / rules text and figures (rate_pca.png, rate_pred_vs_true.png) into --out.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from truesight.analysis import rate_control as rc  # noqa: E402  (imports lpips before sklearn)
from truesight.analysis import recommender as rec  # noqa: E402
from truesight.analysis import rules as rl  # noqa: E402
from truesight.analysis.risk_model import split_by_image  # noqa: E402

SIZE_TARGET, PRECISION_TARGET = 0.80, 0.60


def md(df, fmt="{:.4f}"):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]) + " |")
    return "\n".join(lines)


def figures(df, best_model, test, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coords, var, _, bpp = rc.pca_embedding(df)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.log10(bpp), cmap="viridis", s=14)
    fig.colorbar(sc, label="log10 mean bpp over the six modes")
    ax.set(xlabel=f"PC1 ({var[0]:.0%})", ylabel=f"PC2 ({var[1]:.0%})", title="Image feature space (PCA), coloured by bpp"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "rate_pca.png", dpi=120); plt.close(fig)
    pred, true = np.exp(rc.predict(best_model, test)), test["bpp"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.loglog(true, pred, ".", ms=4, alpha=.5); lim = [true.min() * .8, true.max() * 1.25]
    ax.plot(lim, lim, "k-", lw=1, label="perfect"); ax.plot(lim, [v * 1.1 for v in lim], "k:", lw=.8, label="+-10%"); ax.plot(lim, [v * .9 for v in lim], "k:", lw=.8)
    ax.set(xlabel="measured bpp", ylabel="predicted bpp", title="Size prediction on held-out images"); ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(out / "rate_pred_vs_true.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "experiments" / "step14_intelligence"))
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--targets-per-image", type=int, default=3)
    ap.add_argument("--min-support", type=float, default=0.03)
    ap.add_argument("--min-lift", type=float, default=1.2)
    ap.add_argument("--rule-confidence", type=float, default=0.5)
    ap.add_argument("--quality-bar", type=float, default=rec.QUALITY_BAR, help="LPIPS bar of the oracle (default 0.25)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    a = ap.parse_args()
    data, out, figs = Path(a.data), Path(a.out or a.data), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    table, context, outcomes = (pd.read_csv(data / f) for f in ("mode_table.csv", "context.csv", "outcomes.csv"))
    for c in ("text_ok", "face_ok"):
        if c not in outcomes:
            outcomes[c] = np.nan
    images = np.array(sorted(table.image.unique()))
    tr_i, te_i = split_by_image(images, a.test_frac, seed=0)
    train_imgs, test_imgs = set(images[tr_i]), set(images[te_i])
    train, test = table[table.image.isin(train_imgs)].reset_index(drop=True), table[table.image.isin(test_imgs)].reset_index(drop=True)
    print(f"{len(images)} images ({len(train_imgs)} train / {len(test_imgs)} test), {len(table)} mode rows", flush=True)
    summary = {"images": int(len(images)), "train_images": len(train_imgs), "test_images": len(test_imgs)}
    report = []

    # ------------------------------------------------------------------ 14a rate control
    m_size = rc.fit_models(train, "log_bpp")
    ev = rc.evaluate_models(m_size, test, "log_bpp")
    m_q = rc.fit_models(train, "lpips", names=rc.TREE_MODELS)
    evq = rc.evaluate_models(m_q, test, "lpips")
    tab = pd.DataFrame({"R2 log(bpp)": {k: v["r2"] for k, v in ev.items()}, "size within 10%": {k: v["within_10pct"] for k, v in ev.items()},
                        "R2 LPIPS": {k: evq[k]["r2"] if k in evq else np.nan for k in ev}})
    tab.index.name = "model"
    print("\nRate-control models on held-out images:\n" + tab.round(4).to_string(), flush=True)
    kept = rc.lasso_features(m_size["lasso"], rc.design_matrix(train)[1])
    best = max(ev, key=lambda k: ev[k]["within_10pct"] + 0.01 * ev[k]["r2"])
    sp = rc.size_prediction_summary(m_size[best], test)
    targets = rc.make_targets(test, a.targets_per_image, seed=0)
    per_target, rcs = rc.evaluate_rate_control(m_size[best], test, targets)
    per_target.to_csv(out / "rate_control_targets.csv", index=False)
    summary["rate_control"] = {"models": ev, "lpips_models": evq, "lasso_features": kept, "best_model": best, "size_prediction": sp,
                               "rate_control": rcs, "target_within_10pct_for_80pct": bool(sp["rows_within"] >= SIZE_TARGET)}
    print(f"\nbest model {best}: predicted size within +-10% for {sp['rows_within']:.1%} of (image, mode) rows (target 80%)"
          f"\nrate control: meets {rcs['meets']:.1%} | strictly within +-10% of target {rcs['within_target']:.1%} (six discrete modes can reach "
          f"{rcs['oracle_within']:.1%}) | retry {rcs['retry_rate']:.1%}\nLasso kept: {list(kept)[:8]}", flush=True)
    report += ["## 14a Rate control", md(tab), "", f"Lasso features: {kept}", "", json.dumps({"size_prediction": sp, "rate_control": rcs}, indent=2)]
    figures(table, m_size[best], test, figs)

    # ------------------------------------------------------------------ 14b recommender
    oracle = rec.build_oracle(outcomes, quality_bar=a.quality_bar)
    ctx_tr, ctx_te = context[context.image.isin(train_imgs)], context[context.image.isin(test_imgs)]
    or_tr, or_te = oracle[oracle.image.isin(train_imgs)], oracle[oracle.image.isin(test_imgs)]
    out_te = outcomes[outcomes.image.isin(test_imgs)]
    rec_res = {}
    models = {"knn": rec.KNNRecommender().fit(ctx_tr, or_tr), "tree": rec.TreeRecommender().fit(ctx_tr, or_tr)}
    for name, m in models.items():
        rec_res[name] = rec.evaluate_recommender(m, ctx_te, or_te, out_te, quality_bar=a.quality_bar)
    rec_res["majority_baseline"] = float(max(np.bincount(or_te.oracle, minlength=4)) / len(or_te))
    summary["recommender"] = {**rec_res, "oracle_distribution": or_te.oracle.value_counts().sort_index().to_dict(),
                              "precision_target_met": bool(max(rec_res["knn"]["precision_at_1"], rec_res["tree"]["precision_at_1"]) >= PRECISION_TARGET)}
    print(f"\nrecommender precision@1: kNN {rec_res['knn']['precision_at_1']:.3f} | tree {rec_res['tree']['precision_at_1']:.3f} | "
          f"always-majority {rec_res['majority_baseline']:.3f} (target 0.6)", flush=True)
    tree_rules = models["tree"].export_rules()
    (out / "tree_rules.txt").write_text(tree_rules)
    print("\nDecision-tree rules:\n" + tree_rules, flush=True)
    report += ["## 14b Recommender", json.dumps(summary["recommender"], indent=2, default=float), "", "```", tree_rules, "```"]
    # simulated users: the base recommendation of the kNN on every held-out (image, network) pair
    base, nbytes = [], []
    ob = out_te.set_index(["image", "mode"])["bytes"]
    for net in rec.NETWORKS:
        c = ctx_te.reset_index(drop=True)
        base += list(models["knn"].predict(c, net))
        nbytes += [[ob.loc[(i, m)] for m in rl.MODES] for i in c.image]
    summary["user_history_simulation"] = rec.simulate_users(base, nbytes, seed=0)
    print("user-history simulation:", json.dumps(summary["user_history_simulation"]), flush=True)

    # ------------------------------------------------------------------ 14c failure rules
    th = rec.ItemThresholds.fit(ctx_tr)
    log = rec.failure_log(context, outcomes[outcomes.image.isin(train_imgs)], th)
    # failure_log iterates outcomes rows; use the training outcomes only so that test failures stay unseen
    rules = rl.mine_failure_rules(log, a.min_support, a.min_lift, top=15)
    rules.to_csv(out / "failure_rules.csv", index=False)
    product = rl.rules_to_product_rules(rules, a.rule_confidence)
    lines = rl.format_rules(rules.head(10))
    print(f"\nfailure log: {len(log)} rows, {log.HALLUC.mean():.1%} HALLUC; {len(rules)} rules, {len(product)} product rules")
    print("\n".join(lines) or "(no rules)")
    print("product rules:", [p.name for p in product], flush=True)
    effect = {}
    if product:
        ctx_idx = ctx_te.set_index("image")
        out_idx = out_te.set_index(["image", "mode"])
        for label, prules in (("without_rules", []), ("with_rules", product)):
            fails = n = 0
            r = rec.Recommender(models["knn"], None, prules, th)
            for img, c in ctx_idx.iterrows():
                for net in rec.NETWORKS:
                    mode = r.recommend(c, net).mode
                    o = out_idx.loc[(img, mode)]
                    fails += int(not rec.feasible(o, a.quality_bar))
                    n += 1
            effect[label] = fails / n
    summary["failure_rules"] = {"log_rows": int(len(log)), "halluc_rate": float(log.HALLUC.mean()), "n_rules": int(len(rules)),
                                "top_rules": lines, "product_rules": [{"name": p.name, "min_mode": rl.MODES[p.min_mode], "support": p.support,
                                                                       "confidence": p.confidence, "lift": p.lift} for p in product],
                                "at_least_two_product_rules": bool(len(product) >= 2), "inadequate_recommendation_rate": effect}
    report += ["## 14c Failure rules", "\n".join(lines), "", json.dumps(summary["failure_rules"], indent=2, default=float)]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "report.md").write_text("\n".join(report) + "\n")
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
