"""Run the user-study web page (standard library only, no Gradio).

    python scripts/user_study_app.py --study study --port 8000

Participants open http://<this machine>:8000 on the same network (or expose the port with a tunnel). Answers are appended to
<study>/responses.csv after every pair. Build the pairs first with scripts/make_study_pairs.py.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.analysis.study_server import serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default=str(ROOT / "study"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pairs", type=int, default=20, help="pairs per participant (the guide: 20)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not (Path(a.study) / "pairs.csv").exists():
        sys.exit(f"{a.study}/pairs.csv not found: run scripts/make_study_pairs.py first")
    serve(a.study, a.host, a.port, a.pairs, a.seed)


if __name__ == "__main__":
    main()
