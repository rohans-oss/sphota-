# Running TrueSight / Sphota at full scale on a rented GPU

Everything is driven by one resumable script, `scripts/run_full_pipeline.sh`. On Kaggle (12-hour sessions) use the driver
`scripts/kaggle_run.py` around it: see [KAGGLE.md](KAGGLE.md). This page is for a normal rented machine (RTX 4090, A10, L4, A100).

## 1. Pick a machine

| Need | Value |
|---|---|
| GPU | 1 x 16 GB or more (24 GB is comfortable). One GPU is enough; the stages run one after another. |
| Disk | 120 GB (about 23 GB raw data, extracted images, checkpoints, decoded outputs). |
| OS | Ubuntu 22.04 with a CUDA PyTorch image. `torchac` is compiled on first use, so `build-essential` and `ninja-build` are required. |
| Time | The guide estimates 120-180 T4-class GPU-hours for its original scope. The ReLU and A7 / A8 ablations and the Step 9 warm-up add a little (about 15-25 T4-hours). A 4090 or A100 is several times faster than a T4. This is an estimate: measure the first 1,000 iterations (the logs print the speed) before committing to the budget. |

Use a provider with persistent disk or snapshots, or set `TS_SYNC=hf:<user>/<repo>` (or `dir:<path>`) and every finished stage and a
checkpoint every 30 minutes are uploaded. All training stages resume from `last.pt`; a crash writes the exact state first.

## 2. Set up

```bash
sudo apt-get update && sudo apt-get install -y build-essential ninja-build git
git clone https://github.com/MANOFHATERS/sphota.git && cd sphota
pip install -r requirements.txt          # install the CUDA build of torch first if the image has none
```

## 3. Smoke test (about 15 minutes, a few cents)

Runs every stage with tiny settings into `experiments_smoke/` and `report_smoke/`. It catches environment problems (torchac build,
CUDA, OCR and face-model downloads, disk paths) before the expensive runs. The numbers it produces mean nothing.

```bash
python scripts/prepare_data.py --stage all --workers 8      # data is needed by the smoke run too
SMOKE=1 bash scripts/run_full_pipeline.sh
```

## 4. Full run

```bash
nohup bash scripts/run_full_pipeline.sh > experiments/full_run.log 2>&1 &
tail -f experiments/full_run.log
```

If the machine stops, start the same command again. Finished stages are skipped (`experiments/.done/<stage>`), unfinished training
continues from its checkpoint. To redo a stage delete its marker; to run only some stages use `STAGES="diffusion diffusion_eval"`.
Exit status 75 means "paused at the time budget, run again" (only when `TS_TIME_LIMIT_HOURS` is set).

| Stage | Guide part | What it produces |
|---|---|---|
| setup | - | GPU / torchac check and the unit tests |
| data | Step 1 | 23 GB download; `data/processed/*.txt` splits (train about 4,900 images, val 100, Kodak 24, CLIC valid, 500 text, 500 faces) |
| baselines | Steps 3-5 | JPEG / WebP RD, PCA and VQ baselines, size predictor (CPU) |
| ae | Step 6 | PlainAE for M = 8, 16, 32 (30k iterations each) |
| codec | Step 7 | lambda 0.0130 for 150k iterations, then five fine-tunes of 50k each |
| codec_eval, bitstream | Steps 7-8 | RD curves, BD-rate, MS-SSIM, LPIPS; real `.sphota` files, bit-exact check |
| step9 | Step 9 | unconditional 64x64 DDPM (20k iterations): forward-process figure, sample grid |
| diffusion | Step 10 | conditional U-Net, 200k iterations, batch 32, EMA 0.999 |
| diffusion_eval | Step 10, A3, A6 | CFG and steps sweeps, LPIPS / PSNR on Kodak (24 images) |
| clic_crops | Step 10, section 3.4 | LPIPS and FID on about 2,000 matched 256x256 CLIC crops (resumable; the guide's FID target is measured here) |
| protect | Step 11 | Text Faithfulness Rate and Face Identity Rate with protected regions |
| guidance | Step 12, A4, A5 | tune once on val, freeze, evaluate once on the test sets |
| honesty | Step 13 | eight hallucination detectors, AUROC, learning curves, safe mode |
| intelligence | Step 14 | 1,500 images x 6 modes table with real byte counts, rate control, recommender, failure rules |
| codec_ablation, ablation_eval | A1, A2 | K = 1 and ReLU codecs at lambda 0.0130 and 0.0035; `ablation_k.csv`, `ablation_act.csv` |
| train_ablations | A7, A8 | four 20k-iteration diffusion runs: cosine vs linear schedule, AdamW vs Adam, cosine vs constant LR |
| study_pairs | 8.5 | the images of the user study |
| report | Week 11 | `report/results/` (summary table of the section 3.4 targets, ablation table, faithfulness table, provenance) and `report/figures/` |

## 4b. Plan the evaluation cost first (2 minutes, before the long run)

The U-Net has attention at 1/4 resolution, so the cost of decoding grows with the square of the image size, and the guidance of
Step 12 needs the whole image at once. The evaluation scripts therefore replace an image larger than `--max-pixels` (default
768 x 512, a Kodak image) by the window that covers the most text / faces, and the original used for every metric is that same
window (nothing is resized). `--max-pixels 0` decodes whole images, which is only sensible on a strong GPU.

```bash
python scripts/benchmark_decode.py --diffusion experiments/step10_diffusion/final.pt     # after training; or without --diffusion for a first estimate
```

It times one U-Net call on your GPU and prints hours per evaluation stage for the full-scale settings. If the total is more than you
can afford, lower the limits (text 300 and face 500 images, 20 tuning images, 2,000 crops, 50 / 25 DDIM steps) in the SMOKE / full
blocks at the top of `scripts/run_full_pipeline.sh`.

## 5. After the run

- `experiments/logs/*.log`: one log per stage; `report/results/summary_table.md`: each of the eight targets as MET / NOT MET / not measured.
- `report/report.md` (from `scripts/build_report.py`): the report skeleton with the course-mapping table, all result tables, the limitations
  derived from what was missed and the list of deviations from the guide filled in; the narrative chapters are marked TODO(author).
  `report/weekly_log.md` is the table where a missed target gets its real number and reason. Tags: `bash scripts/tag_week.sh 1` ...
  `eval-freeze`, `v1.0`.
- **User study** (needs 10 or more people): `python scripts/user_study_app.py --study study --port 8000`, participants open the page; then
  `python scripts/analyse_user_study.py` (preference and noticed-change rates with 95% intervals). `make_figures.py` includes it.
- **Freeze, then regenerate:** after the evaluations, `git tag eval-freeze` on a clean commit, then
  `python scripts/make_figures.py --require-freeze` rebuilds every table from the stored outputs and records the commit.
- If a numeric target is missed, write the real number and a one-line reason in `report/weekly_log.md` instead of changing the target.

## 6. Known limits (read before you pay for the full run)

- **Step 15 (command-line tool, API, demo UI) is not built** (you chose to do the UI later).
- **No result in this repository so far comes from a fully trained model.** Every figure reported earlier came from toy runs; the
  pipeline and the tests show the code runs end to end, not that the targets are met.
- **CLIC 2020 test labels are not public.** The CLIC *valid* sets stand in for the test split (recorded in the dataset report).
- **First contact with real models:** the guidance learning rate depends on the trained codec, so `guidance` tunes it on the
  validation split first. If a decode diverges or a number misses the guide's target, expect to adjust settings (for example `--w`,
  the learning rate, more diffusion iterations) and re-run that stage only.
- Evaluation images above 768 x 512 pixels are evaluated on a window of that size (see 4b); CLIC test labels are not public.
- The A7 / A8 runs are 20k iterations each, much shorter than the 200k main run, so only their comparison to each other is meaningful.
- Report writing, slides, the demo video and the git tags of the weekly plan are yours; the code only produces the numbers and tables.
