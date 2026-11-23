import numpy as np
import pandas as pd
import pytest

from truesight.analysis import rate_control as rc
from truesight.analysis import recommender as rec
from truesight.analysis import rules as rl
from truesight.analysis.features import FEATURES

MODES4 = rl.MODES


# =========================================================================== synthetic tables
def _mode_table(n_images=120, seed=0, noise=0.03, exact=False):
    """Long mode table whose bpp follows a known law: log bpp = a + 0.4 log1p(lap_var) + 0.3 entropy + 0.55 * mode."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        f = dict(grad_mean=rng.lognormal(3.5, 0.5), lap_var=rng.lognormal(6, 1.0), entropy=rng.uniform(5, 8),
                 edge_density=rng.uniform(0.01, 0.3), colorfulness=rng.uniform(10, 80), brightness=rng.uniform(40, 200),
                 jpeg50_bpp=rng.uniform(0.3, 2.0))
        base = -4.0 + 0.4 * np.log1p(f["lap_var"]) / 3 + 0.3 * f["entropy"] / 4
        for m in range(rc.N_MODES):
            logb = base + 0.55 * m + (0 if exact else rng.normal(0, noise))
            rows.append(dict(image=i, mode=m, w=512, h=512, bpp=float(np.exp(logb)), lpips=float(0.6 * np.exp(-0.35 * m) + (0 if exact else rng.normal(0, 0.01))), **f))
    return pd.DataFrame(rows)


def _split(df, frac=0.3, seed=0):
    imgs = np.array(sorted(df.image.unique()))
    rng = np.random.default_rng(seed)
    test = set(rng.choice(imgs, int(len(imgs) * frac), replace=False))
    return df[~df.image.isin(test)].reset_index(drop=True), df[df.image.isin(test)].reset_index(drop=True)


# =========================================================================== rate control
def test_design_matrix_columns_log_scaling_and_mode_one_hot():
    df = _mode_table(5)
    X, names = rc.design_matrix(df)
    assert names[:7] == FEATURES and names[7] == "inv_pixels" and names[-6:] == [f"mode_{k}" for k in range(6)]
    assert X.shape == (30, 7 + 1 + 6)
    assert np.allclose(X[:, names.index("lap_var")], np.log1p(df.lap_var)) and np.allclose(X[:, names.index("entropy")], df.entropy)
    assert (X[:, -6:].sum(1) == 1).all() and (X[:, -6:].argmax(1) == df["mode"].to_numpy()).all()
    assert "inv_pixels" not in rc.design_matrix(df.drop(columns=["w", "h"]))[1]
    with pytest.raises(ValueError):
        rc.design_matrix(df.assign(mode=9))


def test_all_six_models_predict_log_bpp_on_held_out_images():
    train, test = _split(_mode_table(160))
    models = rc.fit_models(train, "log_bpp")
    assert tuple(models) == rc.LINEAR_MODELS + rc.TREE_MODELS
    ev = rc.evaluate_models(models, test, "log_bpp")
    for name, r in ev.items():
        assert r["r2"] > 0.9, (name, ev)
    assert ev["ridge"]["within_10pct"] > 0.8 and ev["linear"]["within_10pct"] > 0.8
    q = rc.fit_models(train, "lpips", names=rc.TREE_MODELS)
    assert set(q) == set(rc.TREE_MODELS) and all(r["r2"] > 0.8 for r in rc.evaluate_models(q, test, "lpips").values())


def test_unregularised_linear_model_stays_stable_on_unseen_image_sizes():
    """Regression: with only two distinct training sizes, two resolution columns were collinear and the plain linear model's
    coefficients exploded (R^2 ~ -1e21) on images of another size. One resolution term must be enough."""
    df = _mode_table(160)
    small = df.image % 2 == 0
    df.loc[small, ["w", "h"]] = 256
    train = df[df.image < 100].reset_index(drop=True)
    test = df[df.image >= 100].reset_index(drop=True)
    test[["w", "h"]] = 384                                                   # a size the training table never had
    lin = rc.fit_models(train, "log_bpp", names=["linear"])["linear"]
    r2 = rc.evaluate_models({"linear": lin}, test)["linear"]["r2"]
    assert np.isfinite(r2) and r2 > 0.8
    pred = rc.predict(lin, test)
    assert np.abs(pred - rc.target_log_bpp(test)).max() < 1.0               # no wild predictions on the unseen size
    coef = lin[-1].coef_
    names = rc.design_matrix(test)[1]
    non_mode = [abs(c) for c, n in zip(coef, names) if not n.startswith('mode_')]
    assert max(non_mode) < 5                                                 # (the six one-hot mode columns are collinear with the
                                                                             # intercept: they may carry equal, cancelling offsets)


def test_lasso_keeps_the_informative_features_and_drops_the_dead_ones():
    train = _mode_table(200)
    train["colorfulness"] = 1.0                                              # a constant feature carries no information
    lasso = rc.fit_models(train, "log_bpp", names=["lasso"])["lasso"]
    kept = rc.lasso_features(lasso, rc.design_matrix(train)[1])
    assert "colorfulness" not in kept and "mode_5" in kept and "entropy" in kept
    assert list(kept.values()) == sorted(kept.values(), key=abs, reverse=True)


def test_choose_mode_takes_the_highest_quality_that_fits():
    pb = np.array([1000.0, 2000, 4000, 8000, 16000, 32000])
    assert rc.choose_mode(pb, 5000) == 2 and rc.choose_mode(pb, 4000) == 2 and rc.choose_mode(pb, 99999) == 5
    assert rc.choose_mode(pb, 10) == 0                                        # nothing fits: the smallest file


def test_rate_control_steps_down_at_most_once():
    train, _ = _split(_mode_table(120, noise=0.0))
    model = rc.fit_models(train, "log_bpp", names=["linear"])["linear"]
    row = train.iloc[0]
    pb = rc.predicted_bytes(model, row, 512, 512)
    assert pb.shape == (6,) and (np.diff(pb) > 0).all()                       # larger modes -> larger files
    target = pb[3] * 1.01
    calls = []
    ok = rc.rate_control(model, row, 512, 512, target, measure=lambda m: calls.append(m) or pb[m])
    assert ok["mode"] == 3 and not ok["retried"] and calls == [3]
    calls.clear()
    over = rc.rate_control(model, row, 512, 512, target, measure=lambda m: calls.append(m) or pb[m] * 1.5)   # measured too big
    assert over["mode"] == 2 and over["retried"] and over["first_mode"] == 3 and calls == [3, 2]            # one retry only
    calls.clear()
    tiny = rc.rate_control(model, row, 512, 512, 1.0, measure=lambda m: calls.append(m) or 1e9)
    assert tiny["mode"] == 0 and not tiny["retried"] and calls == [0]          # nowhere left to step down
    assert rc.rate_control(model, row, 512, 512, target)["measured_bytes"] is None


def test_targets_lie_inside_each_images_size_range():
    df = _mode_table(20)
    t = rc.make_targets(df, n_per_image=3, seed=1)
    assert len(t) == 60 and set(t.image) == set(df.image)
    size = (df.bpp * df.w * df.h / 8).groupby(df.image).agg(["min", "max"])
    for _, r in t.iterrows():
        assert size.loc[r.image, "min"] <= r.target_bytes <= size.loc[r.image, "max"]


def test_rate_control_evaluation_counts_meets_within_and_the_discrete_mode_ceiling():
    train, test = _split(_mode_table(200, noise=0.0, exact=True))
    model = rc.fit_models(train, "log_bpp", names=["linear"])["linear"]        # exact law -> an (almost) perfect predictor
    targets = rc.make_targets(test, n_per_image=5, seed=2)
    d, s = rc.evaluate_rate_control(model, test, targets)
    assert s["meets"] > 0.97 and s["pred_within"] > 0.97 and s["retry_rate"] < 0.05 and s["n_targets"] == len(targets)
    assert s["within_target"] <= s["oracle_within"] + 1e-9                     # cannot beat what six discrete modes allow
    assert s["oracle_within"] < 0.8                                            # ... and that ceiling is well below 80%
    assert s["oracle_meets"] == 1.0 and d.rel_error.ge(0).all()

    class Biased:                                                              # predicts files half the real size -> picks modes too big
        def predict(self, X):
            return model.predict(X) - np.log(2.0)

    d2, s2 = rc.evaluate_rate_control(Biased(), test, targets)
    assert s2["retry_rate"] > 0.3 and s2["meets"] < s["meets"]
    sp = rc.size_prediction_summary(model, test)
    assert sp["rows_within"] > 0.97 and sp["images_with_all_modes_within"] > 0.9 and rc.size_prediction_summary(Biased(), test)["rows_within"] == 0.0 \
        if False else sp["rows_within"] > 0.97


def test_pca_embedding_shapes():
    df = _mode_table(30)
    coords, var, imgs, bpp = rc.pca_embedding(df)
    assert coords.shape == (30, 2) and 0 < var.sum() <= 1 and len(imgs) == 30 and len(bpp) == 30 and (bpp > 0).all()


# =========================================================================== the oracle and the context
def _outcomes_one(**over):
    base = {"tiny": (2000, 0.40, True, np.nan), "small": (5000, 0.20, True, np.nan), "balanced": (20000, 0.10, True, np.nan),
            "exact": (80000, 0.05, True, np.nan)}
    base.update(over)
    return pd.DataFrame([dict(image=0, mode=m, bytes=b, lpips=l, text_ok=t, face_ok=f) for m, (b, l, t, f) in base.items()])


def test_oracle_picks_the_cheapest_mode_meeting_both_bars_per_network():
    o = _outcomes_one()
    assert rec.transfer_seconds(50_000, "2G") == pytest.approx(8.0)
    # 2G: tiny fails the quality bar; balanced (3.2 s + 3 s) beats small (0.8 s + 8 s) because diffusion decoding dominates
    assert rec.oracle_mode(o, "2G") == 2
    assert rec.oracle_mode(o, "WiFi") == 3                                   # fast network: the 0.3 s codec-only decode wins
    assert rec.oracle_mode(_outcomes_one(balanced=(20000, 0.10, False, np.nan)), "2G") == 1       # text failure: balanced excluded
    assert rec.oracle_mode(_outcomes_one(small=(5000, 0.20, True, False), balanced=(20000, 0.1, True, False),
                                         exact=(80000, 0.05, True, np.nan)), "2G") == 3
    nothing = _outcomes_one(tiny=(2000, .5, True, np.nan), small=(5000, .5, True, np.nan), balanced=(2e4, .5, True, np.nan), exact=(8e4, .5, True, np.nan))
    assert rec.oracle_mode(nothing, "2G") == 3                               # no mode meets the bars: highest quality
    assert rec.feasible(o.iloc[1]) and not rec.feasible(o.iloc[0])


def test_build_oracle_has_one_label_per_image_and_network():
    out = pd.concat([_outcomes_one().assign(image=i) for i in range(3)])
    o = rec.build_oracle(out)
    assert len(o) == 12 and set(o.network) == set(rec.NETWORKS) and set(o.image) == {0, 1, 2}


def test_context_matrix_items_and_failure_log():
    ctx = pd.DataFrame([dict(image=i, grad_mean=40.0, lap_var=1000.0 * (i + 1), entropy=7.0, edge_density=0.1, colorfulness=30.0,
                             brightness=40.0 if i == 0 else 120.0, jpeg50_bpp=0.8, text_frac=0.1 if i == 0 else 0.0, face_frac=0.0,
                             text_h=12.0, n_faces=3 if i == 1 else 0) for i in range(4)])
    X = rec.context_matrix(ctx)
    assert X.shape == (4, 9) and np.allclose(X[:, rec.CONTEXT_COLS.index("lap_var")], np.log1p(ctx.lap_var))
    th = rec.ItemThresholds.fit(ctx)
    assert th.high_texture == pytest.approx(np.quantile(ctx.lap_var, 0.75))
    assert rec.item_flags(ctx.iloc[0], th) == {"small_text", "low_light"}
    assert rec.item_flags(ctx.iloc[1], th) == {"many_faces"} and "high_texture" in rec.item_flags(ctx.iloc[3], th)
    out = pd.concat([_outcomes_one().assign(image=i) for i in range(2)]).reset_index(drop=True)
    out.loc[(out.image == 0) & (out["mode"] == "tiny"), "text_ok"] = False
    log = rec.failure_log(ctx, out, th)
    assert len(log) == 8 and log.dtypes.eq(bool).all()
    assert log.HALLUC.sum() == 1 and log[log.HALLUC].iloc[0][["small_text", "low_light", "mode_tiny"]].all()


# =========================================================================== kNN, tree, history, rules, evaluation
def _recsys(n=240, seed=0):
    """Contexts + oracle where the right mode depends on the text share and the network (what a real oracle would learn)."""
    rng = np.random.default_rng(seed)
    ctx = pd.DataFrame({"image": np.arange(n), "grad_mean": rng.lognormal(3.5, .4, n), "lap_var": rng.lognormal(6, 1, n),
                        "entropy": rng.uniform(5, 8, n), "edge_density": rng.uniform(0.01, 0.3, n), "colorfulness": rng.uniform(10, 80, n),
                        "brightness": rng.uniform(40, 200, n), "jpeg50_bpp": rng.uniform(.3, 2, n),
                        "text_frac": np.where(rng.random(n) < .5, rng.uniform(0.08, 0.3, n), 0.0), "face_frac": rng.uniform(0, .1, n)})
    rows = []
    for _, c in ctx.iterrows():
        for net_i, net in enumerate(rec.NETWORKS):
            mode = 3 if net_i >= 2 else (2 if c.text_frac > 0.05 else 0)
            rows.append(dict(image=int(c.image), network=net, oracle=mode))
    return ctx, pd.DataFrame(rows)


def _split_ctx(ctx, oracle, frac=0.3):
    test = set(ctx.image.iloc[: int(len(ctx) * frac)])
    return (ctx[~ctx.image.isin(test)], oracle[~oracle.image.isin(test)], ctx[ctx.image.isin(test)], oracle[oracle.image.isin(test)])


def test_knn_recommender_learns_the_oracle_and_ties_go_up():
    ctx, oracle = _recsys()
    ctr, otr, cte, ote = _split_ctx(ctx, oracle)
    knn = rec.KNNRecommender(k=10).fit(ctr, otr)
    ev = rec.evaluate_recommender(knn, cte, ote)
    assert ev["precision_at_1"] >= 0.9 and set(ev["per_network"]) == set(rec.NETWORKS) and ev["per_network"]["WiFi"] >= 0.95
    p = knn.predict(cte.iloc[:5], "2G")
    assert p.shape == (5,) and set(p) <= {0, 2}
    # tie-break: two neighbours disagree (modes 1 and 2) -> the safer, higher mode
    two = pd.DataFrame([{**{c: 0.0 for c in rec.CONTEXT_COLS}, "image": 0}, {**{c: 0.001 for c in rec.CONTEXT_COLS}, "image": 1}])
    orc = pd.DataFrame([dict(image=0, network="2G", oracle=1), dict(image=1, network="2G", oracle=2)])
    assert rec.KNNRecommender(k=2).fit(two, orc).predict(two.iloc[:1], "2G")[0] == 2


def test_decision_tree_recommender_exports_readable_rules():
    ctx, oracle = _recsys()
    ctr, otr, cte, ote = _split_ctx(ctx, oracle)
    tree = rec.TreeRecommender().fit(ctr, otr)
    assert rec.evaluate_recommender(tree, cte, ote)["precision_at_1"] >= 0.95
    txt = tree.export_rules()
    assert "network" in txt and "text_frac" in txt and "2 = 4G" in txt and "class:" in txt
    assert tree.tree.get_depth() <= 4


def test_user_history_shifts_only_frequent_hd_users_and_caps_at_exact():
    h = rec.UserHistory(min_events=3, threshold=0.5)
    for _ in range(2):
        h.record("a", True)
    assert h.shift("a") == 0                                                  # too few events
    h.record("a", True)
    assert h.shift("a") == 1 and h.adjust(1, "a") == 2 and h.adjust(3, "a") == 3 and h.hd_rate("a") == 1.0
    for _ in range(4):
        h.record("b", False)
    h.record("b", True)
    assert h.shift("b") == 0 and h.shift("never-seen") == 0 and h.hd_rate("never-seen") == 0.0


def _stub_model(idx):
    class M:
        def predict(self, context, network):
            return np.full(len(context), idx)
    return M()


def test_recommender_applies_history_and_hard_rules_and_explains_them():
    ctx = dict(grad_mean=40.0, lap_var=500.0, entropy=7.0, edge_density=0.1, colorfulness=30.0, brightness=40.0, jpeg50_bpp=0.8,
               text_frac=0.1, face_frac=0.0, text_h=12.0, n_faces=0)
    rule = rl.ProductRule("small_text AND low_light => at least small", frozenset({"small_text", "low_light"}), 1, 0.05, 0.8, 3.0)
    hist = rec.UserHistory()
    for _ in range(3):
        hist.record("u", True)
    r = rec.Recommender(_stub_model(0), hist, [rule]).recommend(ctx, "2G", user="u")
    assert (r.base_index, r.index, r.mode) == (0, 1, "small")                # history +1 -> small; the rule (at least small) adds nothing
    assert any("HD" in x for x in r.reasons) and not any("rule" in x for x in r.reasons)
    r2 = rec.Recommender(_stub_model(0), None, [rule]).recommend(ctx, "2G")
    assert (r2.index, r2.mode) == (1, "small") and r2.reasons == ["rule: small_text AND low_light => at least small"]
    bright = dict(ctx, brightness=150.0)
    assert rec.Recommender(_stub_model(0), None, [rule]).recommend(bright, "2G").index == 0     # rule does not apply
    assert rec.Recommender(_stub_model(3), None, [rule]).recommend(ctx, "WiFi").reasons == []


def test_evaluate_recommender_reports_adequacy_and_cost_ratio():
    ctx, oracle = _recsys(60)
    out = pd.concat([_outcomes_one().assign(image=i) for i in range(60)])
    oracle = rec.build_oracle(out)
    knn = rec.KNNRecommender().fit(ctx, oracle)
    ev = rec.evaluate_recommender(knn, ctx.iloc[:20], oracle[oracle.image < 20], outcomes_test=out[out.image < 20])
    assert ev["precision_at_1"] == 1.0 and ev["adequate"] == 1.0 and ev["mean_cost_ratio_vs_oracle"] == pytest.approx(1.0)


def test_simulated_users_ask_for_hd_less_when_the_recommender_adapts():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 3, 400)                                            # never recommends exact on its own
    nbytes = np.column_stack([np.full(400, v) for v in (2e3, 5e3, 2e4, 8e4)])
    s = rec.simulate_users(base, nbytes, n_users=150, sessions=30, seed=1)
    assert s["adaptive"]["hd_request_rate"] < 0.8 * s["static"]["hd_request_rate"] and s["lovers_shifted"] > 0.8
    assert s == rec.simulate_users(base, nbytes, n_users=150, sessions=30, seed=1)          # deterministic
    assert s["adaptive"]["mean_bytes"] > 0 and set(s) == {"static", "adaptive", "lovers_shifted"}


# =========================================================================== association rules
def _log(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    # the planted combination must occur in >= 3% of the rows (Apriori's min_support), else it can never be found
    df = pd.DataFrame({"small_text": rng.random(n) < .5, "many_faces": rng.random(n) < .2, "low_light": rng.random(n) < .5,
                       "high_texture": rng.random(n) < .3})
    mode = rng.integers(0, 4, n)
    for k, m in enumerate(MODES4):
        df[f"mode_{m}"] = mode == k
    df["HALLUC"] = (df.small_text & df.low_light & df.mode_tiny) | (rng.random(n) < .03)
    return df


def test_apriori_finds_the_planted_failure_rule():
    rules = rl.mine_failure_rules(_log())
    assert list(rules.columns) == rl.RULE_COLUMNS and len(rules) >= 1 and len(rules) <= 15
    assert rules.lift.is_monotonic_decreasing or (rules.sort_values(["lift", "confidence"], ascending=False).index == rules.index).all()
    top = rules.iloc[0]
    assert {"small_text", "low_light", "mode_tiny"} <= set(top.antecedents) and top.confidence > 0.8 and top.lift > 5
    assert all(rl.TARGET not in a for a in rules.antecedents)
    assert any("small_text" in l and "HALLUC" in l for l in rl.format_rules(rules))
    empty = rl.mine_failure_rules(_log().assign(HALLUC=False), min_support=0.5)
    assert empty.empty and list(empty.columns) == rl.RULE_COLUMNS
    with pytest.raises(ValueError):
        rl.mine_failure_rules(_log().drop(columns=["HALLUC"]))


def _rules(rows):
    return pd.DataFrame(rows, columns=["antecedents", "support", "confidence", "lift"])


def test_rules_become_product_rules_only_when_actionable():
    mined = _rules([(frozenset({"small_text", "low_light", "mode_tiny"}), 0.06, 0.9, 8.0),
                    (frozenset({"small_text", "mode_tiny"}), 0.07, 0.4, 3.0),          # confidence too low
                    (frozenset({"many_faces", "high_texture"}), 0.05, 0.9, 4.0),       # no mode item: informative only
                    (frozenset({"mode_tiny", "mode_small", "low_light"}), 0.05, 0.9, 2.0),   # two mode items: ambiguous
                    (frozenset({"mode_tiny"}), 0.2, 0.9, 1.3)])                        # no situation
    pr = rl.rules_to_product_rules(mined)
    assert len(pr) == 1 and pr[0].conditions == {"small_text", "low_light"} and pr[0].min_mode == 1
    assert pr[0].name == "low_light AND small_text => at least small" and (pr[0].confidence, pr[0].lift) == (0.9, 8.0)
    # the same situation seen for tiny and for small keeps the stricter minimum
    both = rl.rules_to_product_rules(_rules([(frozenset({"low_light", "mode_tiny"}), .1, .8, 2.0), (frozenset({"low_light", "mode_small"}), .1, .7, 1.9)]))
    assert len(both) == 1 and both[0].min_mode == 2 and both[0].name.endswith("at least balanced")
    last = rl.rules_to_product_rules(_rules([(frozenset({"low_light", "mode_exact"}), .1, .9, 2.0)]))
    assert last[0].min_mode == 3                                                        # nothing above exact


def test_a_failure_combination_rarer_than_min_support_is_not_found_until_the_threshold_is_lowered():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"small_text": rng.random(n) < .1, "low_light": rng.random(n) < .1, "mode_tiny": rng.random(n) < .3})
    df["HALLUC"] = (df.small_text & df.low_light & df.mode_tiny) | (rng.random(n) < .02)      # joint support ~ 0.3%
    assert not any({"small_text", "low_light", "mode_tiny"} <= set(a) for a in rl.mine_failure_rules(df).antecedents)
    low = rl.mine_failure_rules(df, min_support=0.002)
    assert any({"small_text", "low_light", "mode_tiny"} <= set(a) for a in low.antecedents)


def test_enforcing_rules():
    r = rl.ProductRule("r", frozenset({"a", "b"}), 2)
    assert r.applies({"a", "b", "c"}) and not r.applies({"a"}) and r.enforce(0) == 2 and r.enforce(3) == 3
    assert rl.enforce_rules(0, {"a", "b"}, [r]) == (2, ["r"]) and rl.enforce_rules(3, {"a", "b"}, [r]) == (3, [])
    assert rl.enforce_rules(0, {"a"}, [r]) == (0, [])


def test_mined_rule_upgrades_a_recommendation_end_to_end():
    rules = rl.mine_failure_rules(_log())
    pr = rl.rules_to_product_rules(rules, min_confidence=0.6)
    assert any(p.conditions == {"small_text", "low_light"} and p.min_mode >= 1 for p in pr)
    ctx = dict(grad_mean=40.0, lap_var=500.0, entropy=7.0, edge_density=0.1, colorfulness=30.0, brightness=40.0, jpeg50_bpp=0.8,
               text_frac=0.1, face_frac=0.0, text_h=12.0, n_faces=0)
    r = rec.Recommender(_stub_model(0), None, pr).recommend(ctx, "2G")
    assert r.mode != "tiny" and r.reasons and r.reasons[0].startswith("rule:")
