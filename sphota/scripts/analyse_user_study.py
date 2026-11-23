"""Analyse the user study: preference for Sphota and rate of noticed changes, with 95% binomial confidence intervals.

    python scripts/analyse_user_study.py --study study --out report/results

Reads <study>/pairs.csv and <study>/responses.csv; writes user_study.csv / user_study.md / user_study.json into --out and
the figure user_study.png into --figs. Prints whether the design of the guide was met (>= 10 people, 20 pairs each).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
from truesight.analysis.user_study import analyse, fmt_markdown  # noqa: E402


def figure(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    y = list(range(len(table)))
    for a, col, title in ((ax[0], "preference", "Prefers Sphota (50% = no preference)"),
                          (ax[1], "notice_any", "Notices a change vs the original")):
        v, lo, hi = table[f"{col}_pct"], table[f"{col}_ci_low"], table[f"{col}_ci_high"]
        a.barh(y, v, xerr=[v - lo, hi - v], color="tab:green" if col == "preference" else "tab:orange", capsize=4)
        a.set_yticks(y); a.set_yticklabels([f"{g} (n={n})" for g, n in zip(table.group, table.pairs)])
        a.set_xlim(0, 100); a.set_title(title, fontsize=10); a.grid(alpha=.3, axis="x")
    ax[0].axvline(50, c="k", lw=1, ls="--")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default=str(ROOT / "study"))
    ap.add_argument("--out", default=str(ROOT / "report" / "results"))
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    a = ap.parse_args()
    s = Path(a.study)
    if not (s / "responses.csv").exists():
        sys.exit(f"{s}/responses.csv not found: no answers collected yet (python scripts/user_study_app.py)")
    table, design = analyse(pd.read_csv(s / "responses.csv"), pd.read_csv(s / "pairs.csv"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "user_study.csv", index=False)
    (out / "user_study.md").write_text(fmt_markdown(table, design) + "\n", encoding="utf-8")
    (out / "user_study.json").write_text(json.dumps({"design": design, "table": table.to_dict("records")}, indent=2, default=float))
    figure(table, Path(a.figs) / "user_study.png")
    print(fmt_markdown(table, design))


if __name__ == "__main__":
    main()
