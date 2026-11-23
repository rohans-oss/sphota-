# Fitting the run into about 24 GPU-hours (`TS_BUDGET=24h`)

For two 12-hour Kaggle sessions. **Every stage, ablation and evaluation still runs**; only three sizes shrink, and each is reported as such:

| Setting | Full run | `TS_BUDGET=24h` |
|---|---|---|
| Diffusion iterations (cosine LR decays to 0 at the total) | 200,000 | 60,000 |
| Training ablations A7 / A8 (4 runs) | 20,000 each | 5,000 each |
| Evaluation samples (`N_IMG`, `CR_MAX`, `N_TBL`) | 500 / 2,000 / 1,500 | 200 / 500 / 600 |

```bash
TS_BUDGET=24h nohup bash scripts/run_full_pipeline.sh > experiments/full_run.log 2>&1 &
# any single value can be overridden: TS_D_IT, TS_ABL_IT, TS_D_SNAPSHOTS, TS_ABL_SNAPSHOTS
```

## What the shorter runs cost: measured, not guessed

The preset keeps checkpoints that are never overwritten:

* main model: `snapshot_30000.pt` and `snapshot_60000.pt`, written during training. For a run that is already past 30k, first run
  `python scripts/snapshot_checkpoint.py experiments/step10_diffusion/last.pt` to keep the current state **before** resuming with the new total.
* ablations: `snapshot_2500.pt` and `snapshot_5000.pt` for each of the four variants, evaluated automatically into
  `experiments/step10_ablation/summary_by_step.csv` (is the ranking at 2.5k the same as at 5k?).

Compare the main-model checkpoints on the same ~30 Kodak images (LPIPS / PSNR, paired bootstrap CI, codec-only reference row):

```bash
python scripts/compare_checkpoints.py --checkpoints experiments/step10_diffusion/snapshot_30000.pt \
    experiments/step10_diffusion/snapshot_60000.pt --images 30 --steps 25
```
