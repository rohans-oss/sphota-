"""Text Faithfulness Rate, Face Identity Rate and hallucination AUROC in one table, with 95% confidence intervals.

    python scripts/eval_faithfulness.py                      # reads experiments/, writes report/results/faithfulness.*

The measurements themselves come from the evaluation scripts (scripts/eval_sphota.py --set text --tfr, --set face --fir, and
scripts/eval_hallucination.py); this script gathers their pass counts for every decoding variant (plain diffusion, ROI only,
guidance only, full Sphota) into the headline table of section 8, adds the exact (Clopper-Pearson) interval of each rate and
the AUROC of the eight detectors, and draws report/figures/faithfulness.png. It recomputes nothing, so it is cheap and
deterministic; a missing evaluation shows up as an empty table, not as a number.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.analysis.report import auroc_rows, faithfulness_rows, to_markdown  # noqa: E402

ORDER = ["coarse", "plain", "roi_only", "guided_only", "sphota"]
LABEL = {"coarse": "coarse codec", "plain": "plain diffusion", "roi_only": "ROI only", "guided_only": "guidance only", "sphota": "Sphota"}


def figure(faith, auroc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (metric, setname) in zip(ax[:2], (("TFR", "text"), ("FIR", "face"))):
        d = faith[(faith.metric == metric) & (faith.set == setname)]
        d = d.set_index("method").reindex([m for m in ORDER if m in set(d.method)])
        if len(d):
            a.bar(range(len(d)), d.rate_pct, yerr=[d.rate_pct - d.ci_low, d.ci_high - d.rate_pct], capsize=4,
                  color=["tab:green" if m == "sphota" else "tab:gray" for m in d.index])
            a.set_xticks(range(len(d))); a.set_xticklabels([LABEL[m] for m in d.index], rotation=20, fontsize=8)
        a.set(ylim=(0, 100), ylabel="% passing", title=f"{metric} on the {setname} set (95% CI)"); a.grid(alpha=.3, axis="y")
    if len(auroc):
        ax[2].barh(auroc.detector[::-1], auroc.auroc[::-1], color="tab:blue")
        ax[2].axvline(0.75, c="r", ls="--", lw=1)
    ax[2].set(xlim=(0.4, 1.0), title="Hallucination AUROC (held-out images; target 0.75)"); ax[2].grid(alpha=.3, axis="x")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=str(ROOT / "experiments"))
    ap.add_argument("--out", default=str(ROOT / "report" / "results"))
    ap.add_argument("--figs", default=str(ROOT / "report" / "figures"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    faith, auroc = faithfulness_rows(a.exp), auroc_rows(a.exp)
    faith.to_csv(out / "faithfulness.csv", index=False); auroc.to_csv(out / "auroc.csv", index=False)
    md = "## Faithfulness rates (95% Clopper-Pearson intervals)\n\n" + to_markdown(faith, "{:.1f}") + \
         "\n## Hallucination detectors (AUROC on held-out images)\n\n" + to_markdown(auroc, "{:.3f}")
    (out / "faithfulness.md").write_text(md, encoding="utf-8")
    figure(faith, auroc, Path(a.figs) / "faithfulness.png")
    print(md)
    if not len(faith) and not len(auroc):
        print("nothing measured yet: run eval_sphota.py (--tfr / --fir) and eval_hallucination.py first")


if __name__ == "__main__":
    main()
