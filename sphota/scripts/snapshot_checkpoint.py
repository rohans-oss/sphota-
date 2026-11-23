"""Keep the CURRENT state of a running diffusion training as a named snapshot, before a resume changes the schedule.

    python scripts/snapshot_checkpoint.py experiments/step10_diffusion/last.pt            # -> snapshot_<step>.pt beside it

Writes snapshot_<step>.pt (raw + EMA weights and config, the final.pt format, no optimiser state) so the model "as of step N" can be
evaluated later with scripts/compare_checkpoints.py, even after the run has continued to a larger step.
"""
import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    s = torch.load(src, map_location="cpu")
    dst = src.with_name(f"snapshot_{s['step']}.pt")
    tmp = Path(str(dst) + ".tmp")
    torch.save({"unet": s["unet"], "ema": s["ema"], "step": s["step"], "config": s["config"]}, tmp)
    tmp.replace(dst)
    print(f"{dst} (step {s['step']})")


if __name__ == "__main__":
    main()
