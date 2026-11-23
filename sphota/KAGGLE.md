# Training the full project on Kaggle (12-hour sessions, crash-proof)

Kaggle stops a notebook after 12 hours and wipes its disk. This project is set up so that **nothing is lost** and **every new
session continues exactly where the last one stopped**, instead of starting again:

| Situation | What happens |
|---|---|
| The 12 hours are nearly used (default budget 11.5 h) | The running trainer saves its exact state (weights, optimiser, EMA, iteration), the driver uploads everything and exits with status 75 ("paused"). Start the notebook again: it downloads the state and continues. |
| Your code has a bug and crashes after 6 hours | The trainer catches the crash, writes the exact state of the last completed step to `last.pt` *before* dying, and the driver uploads it. Fix the bug, push the fix, start the notebook: it continues from that step, not from zero. |
| Kaggle kills the machine without warning | A checkpoint is written every 20 minutes and uploaded every 30 minutes, so you lose at most about 30 minutes. |
| A stage already finished | It is marked done and skipped in later sessions, so no result is ever recomputed. |

Where the state lives: a **private Hugging Face repo** (checkpoints, logs, results, stage markers). Training data lives in a
second private Hugging Face *dataset* repo (or an attached Kaggle Dataset), so a new session does not re-download 23 GB from the
original sites.

## One-time setup

1. **Hugging Face**: create an account, then a token with *write* access (Settings > Access Tokens). Nothing else to create: the two
   private repos are made automatically on first use.
2. **Kaggle notebook**: New Notebook, Accelerator = GPU (T4 x2 or P100), Internet = On, Add-ons > Secrets > add `HF_TOKEN` = your token.
   Choose T4 x2: the pipeline runs independent jobs on both GPUs (see below).
3. Put your names in the cell below (`YOUR_USER`), and keep this as the notebook's only cell.

```python
import os, subprocess
if not os.path.exists("sphota"):
    subprocess.run(["git", "clone", "https://github.com/MANOFHATERS/sphota.git"], check=True)
!python sphota/scripts/kaggle_run.py --sync hf:YOUR_USER/sphota-run --data-repo YOUR_USER/sphota-data --data-dir /kaggle/temp/sphota_data --weekly-quota 30
```

`--weekly-quota 30` (your weekly GPU hours) is what protects the run from the quota ending mid-session; see "Weekly GPU quota" below.

(If the GitHub repository is private, clone with a token the same way, from a Kaggle secret.)

`--data-dir /kaggle/temp/...` keeps the 23 GB of images off `/kaggle/working`, which is limited to about 20 GB. Check `!df -h` once;
if `/kaggle/temp` is smaller than you need, tell me and I will change the layout.

## Every session

Press **Save Version > Save & Run All** (runs unattended for up to 12 hours) or run the cell interactively. When the cell ends:

* `ALL STAGES FINISHED.`  you are done.
* `PAUSED ... the state is saved.`  start the next session (same cell, nothing to change).
* `CRASHED ...`  read the error above it, fix the code (edit, commit, `git push`), start the next session. The driver runs
  `git pull` first, so the fix is picked up.

Restart time: fetching the run state (about 1-2 GB) and the code takes about a minute; missing Python packages are installed once per
session (1-3 minutes); the data restore takes a few minutes from the Hugging Face dataset and is instant from an attached Kaggle
Dataset. Do not expect exactly 2 minutes, expect roughly 3-8.

The first session also does the one-time work: the 23 GB download from the original dataset sites and its upload to your data repo.
That first session may end with `PAUSED` before training starts; the data is then ready for all later sessions.

## Automatic restarts (no waiting between sessions)

Everything is saved to Hugging Face by itself (an upload every 15 minutes, a checkpoint about every 9, a final upload when a session ends).
With `--chain` the **next notebook also starts by itself** when a session ends, using the official `kaggle kernels push` (it makes a new
version of the notebook and runs it). Three notebooks take part (`truesight/chain.py`):

| Notebook | Accelerator | What it does |
|---|---|---|
| `sphota-data` | CPU (uses **no** GPU quota) | downloads and prepares the data, uploads it to your Hugging Face dataset repo, then starts `sphota-train` |
| `sphota-train` | GPU T4 x2 | trains; when a session pauses at its time budget (state saved) it starts the next `sphota-train` session at once; when everything is done it stops |
| `sphota-wait` | CPU | started when the weekly GPU quota is used up: sleeps until the quota reset (up to 11 h per notebook, then another waiter), then starts `sphota-train` |

A notebook cannot change its accelerator while it runs, which is why the data step and the training are two notebooks.

**One-time setup**
1. Kaggle > Settings > API > create a token. Put `KAGGLE_USERNAME` and `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`) in the secrets of each notebook (Add-ons >
   Secrets), together with `HF_TOKEN` (and `GITHUB_TOKEN` if the repository is private). A secret must be ticked for the notebook that uses it.
2. On your computer: `pip install kaggle`, put the token in `~/.kaggle/kaggle.json`, then start the chain (PowerShell, one line per command):
   ```
   python scripts/kaggle_chain.py start --kind cpu_data --owner YOUR_KAGGLE_USER -- --sync hf:YOU/sphota-run --data-repo YOU/sphota-data --data-dir /kaggle/temp/sphota_data --stages data --weekly-quota 30 --quota-reset-dow 0 --chain
   ```
   (In Git Bash put `MSYS_NO_PATHCONV=1` in front, or it rewrites `/kaggle/...`.) `--quota-reset-dow` is the weekday of your quota reset (Monday = 0, UTC);
   look at the date on the Kaggle quota page. Without it the quota is counted over a rolling 7 days and the waiter cannot be used.
3. The first time, `sphota-data` has no secrets attached yet and stops at once with "HF_TOKEN not found" (it wastes nothing). Open it in the editor,
   attach the secrets, **Save Version > Save & Run All** once. The same happens once for `sphota-train` when it is first started by the data notebook:
   attach its secrets and run it once. After that the notebooks restart themselves. (Whether Kaggle keeps secrets attached across versions pushed
   by the API is not documented: if a chained start ever fails with "HF_TOKEN not found", attach again and run once more.)
4. To skip the data step (data already uploaded), start `--kind gpu` instead, without `--stages data`.

**Safeguards.** At most 40 chained starts (`--max-chain`); a crash never chains; a paused session shorter than 6 minutes does not chain (no loops);
the quota is checked again before every GPU start (the ledger since the last reset, or the explicit `--quota-hours` minus what this session used);
a push that Kaggle rejects stops the chain without error. **Kill switch:** create an empty file named `.stop_chain` in your Hugging Face run repo (Files >
Add file) and the chain stops at its next step; `python scripts/kaggle_chain.py stop-help` repeats this. To stop right now, stop the running notebook.

**Not verified here** (it needs a real Kaggle account; the tests replace the kaggle CLI with a recorder): that a running notebook may push another
notebook, that secrets stay attached across API versions, and that `machine_shape: NvidiaTeslaT4` selects "GPU T4 x2" (a three-year-old kaggle-cli issue
says so; if the new notebook starts on a P100, change `--machine-shape` or set the accelerator by hand once). Watch the first data-to-GPU handover and check the new notebook's accelerator before leaving it unattended. Both notebooks briefly overlap at a handover (a few
seconds of double GPU time).

## Weekly GPU quota (the driver plans around it)

Kaggle ends a session without warning when the weekly GPU quota is used up, and no API tells a notebook how much is left. So the
driver plans the budget itself and never starts a session it cannot finish cleanly:

| Flag | Meaning |
|---|---|
| `--weekly-quota 30` | Keeps a ledger (`quota/<session>.json`, uploaded with the run state) and counts the hours of the last 7 days: budget = min(11.5, 30 - used - 0.5). A killed session counts as having used its whole reservation, so the count is never too low. |
| `--quota-hours R` | R = the hours the Kaggle page shows as left right now: budget = min(11.5, R - 0.5). Use it instead, or as a correction if the ledger is off. |
| none | Works, but prints a warning: the quota is invisible to the driver. |

Trainers pause and save when the budget is used up, so the quota is never hit mid-run; the 0.5 h margin is for the final upload.
With less than 15 minutes of budget the session does not start (`NO_QUOTA`, exit 76, nothing touched). If the quota is wrong anyway,
checkpoints are written every 9 minutes and uploaded every 15 (`--sync-every`), so at most about 15 minutes are lost.

## Faster, with exactly the same data, models and results

Three things shorten the run without touching what is trained (all are on by default; `scripts/profile_train.py` measures them on your GPU in
about 5 minutes, run it once before the long run):

1. **Decoded-image cache** (`scripts/build_cache.py`, run automatically by the pipeline). Every training sample used to decode a full 2K
   PNG just to keep one crop. The images are now decoded once into a raw uint8 file and the trainers read crops from it. The crops are
   the same: tests check that the tensors are identical to the PNG path for the same random state. It needs about 41 GB for the 4,909
   images on the data disk (`--data-dir /kaggle/temp/...`). If the disk is smaller, as many images as fit are cached and the rest are
   decoded from their PNG as before: still the same data, just less speed-up. `NO_CACHE=1` skips it, `TS_IMAGE_CACHE=off` ignores it.
   The cache is rebuilt in each new session (a few minutes); it is not uploaded.
2. **Speed settings** (`truesight/speed.py`): `cudnn.benchmark`, `channels_last` for the fp16 U-Net (off for the fp32 codec, where it
   has no Tensor-Core benefit), and one fused call for the EMA update. Float results are the same apart from rounding noise.
   `TS_CUDNN_BENCHMARK=0` / `TS_CHANNELS_LAST=none|unet|all` change them if the profile shows a setting does not help on your GPU.
3. **One job graph over both GPUs** (`scripts/train_phase.py`, used with 2+ GPUs). A GPU never waits while some job is ready:

| Job | Needs | Runs |
|---|---|---|
| lambda 0.0130 base (150k its) | nothing | GPU 0 at once, alone: nothing can share it |
| K=1 ablation base, ReLU ablation base | nothing | GPU 1 at once (used to idle), then their fine-tunes |
| lambda 0.0018 fine-tune, then **diffusion** | the 0.0130 base; then the 0.0018 codec only | diffusion starts the moment the 0.0018 codec exists, not after all six |
| other 4 fine-tunes, `train_ablations`, `ae`, `step9` | the base / the 0.0018 codec / nothing | on whichever GPU is free |
| `baselines` | nothing | on the CPU beside the GPUs |
| after training: `codec_eval -> bitstream -> ablation_eval -> diffusion_eval -> clic_crops -> protect` and `guidance -> honesty -> study_pairs -> intelligence` | as named | two chains side by side, then `report` |

With one GPU everything runs one after the other, in the same order. If one job fails, the others get SIGTERM and save their exact state, so
no work is lost. Each job writes only its own folders; its console output goes to `experiments/logs/job_<name>.out`. Whether a "T4 x2"
session costs one or two sessions of quota is something to check on your account after the first run: look at the quota counter.

## Round two: more speed with the same data, models and quality

Based on the research report (`reports/Faster training without quality loss.md`). Nothing here reduces data, iterations or model size.
What can change numerics is **off until a parity check passed**.

**Always on (exact or float noise only):**
- **Tiled cache.** The decoded-image cache is stored as 128 x 128 tiles (a crop touches at most 4 contiguous tiles instead of 128 scattered
  row reads) and memory-mapped with `MADV_RANDOM`, because 41 GB does not fit in the ~29 GB of RAM and part of every read goes to the disk.
  Same pixels, same crops (tests compare to the PNG path). An old (untiled) cache is rebuilt automatically.
- **Flip and float conversion on the GPU.** The loader returns uint8 crops (4x less host-to-device traffic, no float work on the busy CPU
  cores); the trainer converts and flips each sample with probability 0.5 on the GPU. Same distribution.
- **Fused Adam / AdamW** (`TS_FUSED_OPTIM=0` to turn off), pinned memory and `prefetch_factor=4` (`TS_PREFETCH`).
- **Data-wait meter.** Every log line of `train_codec` / `train_diffusion` ends with `data-wait N%`: the share of time the GPU waited for a
  batch. Near 0%: the loader is fine. Above ~15%: the run is data-bound. `scripts/profile_train.py` prints a DATA-BOUND / GPU-BOUND verdict.
- **Evaluation reuse (exact):** the plain / ROI-only / guided decodes of one image share the denoising steps before their guidance starts
  (`PrefixCache`; `tune_guidance.py` and `eval_sphota.py` use it), the first denoising step is shared across the CFG-scale and step-count
  sweeps (`FirstStepCache`), and OCR / face-detector / face-embedding results of the unchanged original are computed once instead of once per
  method. Tests check that every reused decode equals the from-scratch decode.

**Only after a parity check (otherwise everything stays fp32 exactly as in the guide):**
- **fp16 codec convolutions** (`TS_CODEC_AMP`, default `auto` in the pipeline). The plain convolutions run in fp16; GDN, the last layer of every
  stack (so the latent that is rounded, the mixture parameters and the reconstruction keep fp32 precision), the entropy model, likelihoods, rate,
  MSE and the auxiliary loss stay fp32, with a loss scaler that counts skipped steps. Run `STAGES=amp_parity` once (about an hour: fp32 vs fp16,
  2 seeds each, Kodak-quality evaluation in fp32). It writes `experiments/amp_parity.json`; `amp_ok` is true only if the PSNR / bpp differences
  are within the seed-to-seed spread, no more than 1% of steps were skipped and nothing went non-finite. It also prints the measured speed-up. If
  it fails, training stays fp32 automatically. The research found no published rate-distortion evidence either way, so for the final decision
  also compare a half-length (75k) run at one lambda (`train_codec --amp 1` against `--amp 0`).
- **fp16 U-Net decoding and batched guidance passes** (`TS_EVAL_AMP`, `TS_EVAL_BATCH_CFG`, default `auto`). The `eval_parity` stage (5-10 minutes,
  in the default run, before the evaluation) decodes a few images with the fp32 reference and the candidates and enables each only if it matches
  (PSNR >= 60 dB for the batched passes; for fp16: >= 35 dB and 10 dB above the seed-to-seed difference, paired LPIPS change <= 0.01).
  The bin-consistency guidance itself (the gradient steps through the codec encoder) always stays fp32.
- **fp16 frozen codec inside diffusion training** (`TS_COND_AMP=1`, default off): it can round a latent differently from the fp32 decoder that
  makes the same image at test time, so it is never enabled automatically.

**Deliberately not done** (the research found no gain or a risk): `torch.compile` (0-10% typical, regressions on T4, changes the RNG stream),
CUDA graphs, DDP across the two T4s (slower than independent jobs in the only Kaggle measurement), gradient accumulation, changes to the
training objective (Min-SNR, v-prediction). Splitting decoding and metrics into separate stages saves no GPU time by itself, so it was not built.

## Two notebooks on the same run repo

Safe, as long as they do different work: pass `--claims` to both and give each its own `STAGES` (`--stages "diffusion"` in one, `--stages "codec_ablation"` in
the other). With `--claims`, every stage is claimed first (`.running/<stage>.json`, refreshed by each upload, ignored after 45 minutes without a
refresh; off by default because the claim of a cancelled session would block a single chain for 45 minutes), so a stage that is running elsewhere or finished is skipped. A push sends only the files that session changed since its
last pull, so a session can never overwrite a newer checkpoint from the other with a stale copy. A stage whose input is made by the
other notebook waits (exit 75) until it exists. Both notebooks count against the same account quota.

## What runs, in order (resume-able stages)

`setup data baselines ae codec codec_eval bitstream step9 diffusion diffusion_eval clic_crops protect guidance honesty intelligence
codec_ablation ablation_eval train_ablations study_pairs report`. Run a subset with `--stages "diffusion diffusion_eval"`; try everything with tiny
settings first with `--smoke` (about 15 minutes of GPU). Each finished stage is uploaded immediately.

## Honest limits

* **Kaggle GPU quota.** Kaggle limits GPU hours per week (check the number shown on your account; it has been about 30 h). The guide's
  budget is 120-180 T4-hours, so expect several weeks of sessions. Training alone is the large part; only the long runs
  (`codec`, `diffusion`, `codec_ablation`, `train_ablations`) matter for the quota.
* **Resume is exact in state, not bit-identical in data order.** Weights, optimiser, EMA and the step counter come back exactly;
  the random crops after a restart are drawn with a seed that depends on the step, so the run is statistically the same as an
  uninterrupted one but not byte-for-byte identical.
* **Check the evaluation time before the long run:** `python scripts/benchmark_decode.py` prints the projected hours of every evaluation
  stage on your GPU (see CLOUD_RUNBOOK.md, 4b). Evaluation images are capped at 768 x 512 pixels per image for this reason.
* **A stage that is not a trainer restarts from its beginning if it is cut off** (the CLIC crop stage is the exception: it keeps its batches), for example an evaluation that needs 40 minutes
  and is interrupted at minute 39. The driver avoids this by not starting a stage when too little session time is left.
* **If the whole Kaggle kernel is killed** (not the Python process), the last upload can be up to 30 minutes old.
* **Kaggle limits change** (session length, quota, disk, scheduled runs). Re-check them before a long run.
* The user study needs real people and a machine they can reach (`python scripts/user_study_app.py`), so it is not part of the unattended run.
