"""Step 9 warm-up: unconditional 64x64 DDPM + forward-process figure. See truesight/diffusion/uncond.py for the options.

    python scripts/step9_ddpm.py                      # ~20k iterations as in the guide
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truesight import runtime  # noqa: E402
from truesight.diffusion.uncond import main  # noqa: E402

if __name__ == "__main__":
    runtime.run_cli(main)
