"""Write report/report.md: the report skeleton with every table filled from the measured results.

    python scripts/make_figures.py          # first: tables from experiments/
    python scripts/build_report.py          # then: report/report.md

The course-mapping table, the results tables, the limitations (derived from what is missing or missed) and the list of deviations
from the guide are generated; the narrative chapters are left as TODO(author) for you to write. Re-run after every evaluation so
the numbers in the report are never typed by hand. Also creates report/weekly_log.md from its template if it does not exist yet.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.analysis.report_doc import build_report  # noqa: E402

WEEKLY_LOG = """# Weekly log

Rule of the guide: if a numeric target is missed, write the real number and a one-line reason here, then continue.
Fill one row per measured target when the number exists (copy it from `report/results/summary_table.md`).

| Week | Target | Real number | Met? | One-line reason if missed |
|---|---|---|---|---|
| 1 | Train set has at least 3,000 images | 4,909 (data/processed/train.txt) | yes | |
| 2 | Best size predictor CV R2 >= 0.6 | | | |
| 3 | Kodak PSNR above JPEG at the same estimated bpp (60k-iteration checkpoint) | | | |
| 4 | BD-rate vs JPEG (PSNR) <= -30% on Kodak | | | |
| 4 | Real file bpp within 3% of the estimate (+ header) | | | |
| 4 | Encode + decode of one Kodak image under 10 s on CPU | | | |
| 5 | Unconditional samples are structure, not noise or flat grey | | | |
| 6 | Diffusion LPIPS at least 25% lower than the coarse decode at equal bpp | | | |
| 7 | ROI bits as a share of the total on the text set | | | |
| 8 | Consistency rate of the guided decode >= 0.95 | | | |
| 8 | Text Faithfulness Rate >= 2x plain diffusion | | | |
| 8 | Face Identity Rate >= 1.5x plain diffusion | | | |
| 8 | LPIPS degradation versus unguided diffusion <= about 10% | | | |
| 9 | Best hallucination detector AUROC >= 0.75 | | | |
| 10 | Rate control within +-10% of the target size for 80% of held-out images | | | |
| 10 | Recommender precision@1 >= 0.6 against the oracle | | | |
| 11 | FID on CLIC crops lower than JPEG, WebP and the VAE codec | | | |
| 11 | User study: at least 10 people x 20 pairs, 95% interval reported | | | |

## Notes per week

(Write what you did, what went wrong and what you decided. One short paragraph per week.)

"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "report" / "results"))
    ap.add_argument("--out", default=str(ROOT / "report" / "report.md"))
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(a.root, a.results), encoding="utf-8")
    log = out.parent / "weekly_log.md"
    if not log.exists():
        log.write_text(WEEKLY_LOG, encoding="utf-8")
        print(f"created {log}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
