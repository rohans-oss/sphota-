"""Failure mining with association rules (Step 14c) and the product rules they become.

A failure log has one row per decoded image (per mode) with boolean item columns, for example
    small_text, many_faces, low_light, high_texture, mode_tiny, mode_small, ..., HALLUC
Apriori finds frequent item sets, ``association_rules`` turns them into rules and we keep those that predict HALLUC, ranked by
lift then confidence, e.g.  {small_text, low_light, mode_tiny} -> {HALLUC}.  A rule whose antecedent contains a mode item
becomes a hard product rule: "small text in low light: never use Tiny" (ProductRule.min_mode), enforced by the recommender.
"""
from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

MODES = ("tiny", "small", "balanced", "exact")            # quality order: tiny = smallest file, exact = codec only
TARGET = "HALLUC"
CONTEXT_ITEMS = ("small_text", "many_faces", "low_light", "high_texture")
RULE_COLUMNS = ["antecedents", "support", "confidence", "lift"]


def mode_item(mode):
    return f"mode_{mode}"


def mine_failure_rules(df, min_support=0.03, min_lift=1.2, top=15, target=TARGET):
    """Rules {items} -> {target} from a boolean failure log, sorted by lift then confidence. Empty DataFrame (with the
    RULE_COLUMNS) when nothing qualifies. ``df`` must contain the ``target`` column. A rule can only be found if its whole
    antecedent + target occur together in at least ``min_support`` of the rows: for rarer failure combinations lower
    ``min_support`` (and expect more spurious rules) or log more decodes."""
    items = df.astype(bool)
    if target not in items.columns:
        raise ValueError(f"failure log has no {target!r} column")
    freq = apriori(items, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)
    try:
        rules = association_rules(freq, num_itemsets=len(items), metric="lift", min_threshold=min_lift)
    except TypeError:                                                  # older mlxtend without num_itemsets
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    if rules.empty:                                                    # frequent items but no rule passes the lift threshold
        return pd.DataFrame(columns=RULE_COLUMNS)
    rules = rules[rules["consequents"].apply(lambda s: s == frozenset({target}))]
    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)
    return rules.sort_values(["lift", "confidence"], ascending=False)[RULE_COLUMNS].head(top).reset_index(drop=True)


@dataclass(frozen=True)
class ProductRule:
    """If every item in ``conditions`` holds for an image, the mode must be at least ``min_mode`` (an index into MODES)."""
    name: str
    conditions: frozenset
    min_mode: int
    support: float = 0.0
    confidence: float = 0.0
    lift: float = 0.0

    def applies(self, items):
        return self.conditions <= set(items)

    def enforce(self, mode_idx):
        return max(mode_idx, self.min_mode)


def rules_to_product_rules(rules, min_confidence=0.5, modes=MODES):
    """Turn mined rules into hard constraints. Only rules whose antecedent contains a mode item are actionable: the other items
    are the situation, the mode item is what to avoid, so the new minimum is the next mode up. For {small_text, low_light,
    mode_tiny} -> HALLUC with confidence >= ``min_confidence``: 'small_text AND low_light  =>  at least Small'.
    The same situation found for several modes keeps the strictest (highest) minimum. Returns a list of ProductRule."""
    best = {}
    for _, r in rules.iterrows():
        ante = set(r["antecedents"])
        found = [m for m in modes if mode_item(m) in ante]
        if len(found) != 1 or r["confidence"] < min_confidence:
            continue
        situation = frozenset(ante - {mode_item(found[0])})
        if not situation:
            continue
        new_min = min(modes.index(found[0]) + 1, len(modes) - 1)
        if situation not in best or new_min > best[situation].min_mode:
            label = " AND ".join(sorted(situation))
            best[situation] = ProductRule(f"{label} => at least {modes[new_min]}", situation, new_min,
                                          float(r["support"]), float(r["confidence"]), float(r["lift"]))
    return sorted(best.values(), key=lambda p: (-p.lift, p.name))


def enforce_rules(mode_idx, items, product_rules):
    """Apply every matching product rule; returns (mode index, list of rule names that changed the result)."""
    applied = []
    for r in product_rules:
        if r.applies(items):
            new = r.enforce(mode_idx)
            if new != mode_idx:
                applied.append(r.name)
                mode_idx = new
    return mode_idx, applied


def format_rules(rules):
    """Human-readable lines 'a AND b => HALLUC  (support, confidence, lift)'."""
    return [f"{' AND '.join(sorted(r['antecedents']))} => {TARGET}   support {r['support']:.3f}  confidence {r['confidence']:.2f}  lift {r['lift']:.2f}"
            for _, r in rules.iterrows()]
