#!/usr/bin/env bash
# Full-scale run of the whole project as written in the build guide, resumable across crashes and 12-hour sessions.
#
#   bash scripts/run_full_pipeline.sh                  # everything, skipping stages already finished
#   STAGES="data baselines" bash scripts/run_full_pipeline.sh
#   SMOKE=1 bash scripts/run_full_pipeline.sh          # same commands with tiny settings (~15 min): run this first
#   python scripts/kaggle_run.py --sync hf:USER/REPO --weekly-quota 30    # on Kaggle: restores state, runs, saves state (see KAGGLE.md)
#
# Stages, in order:
#   setup data baselines ae codec codec_eval bitstream step9 diffusion diffusion_eval clic_crops protect guidance honesty
#   intelligence codec_ablation ablation_eval train_ablations study_pairs report
# (codec = the six lambdas; codec_ablation = K=1 and ReLU runs for A1 / A2; train_ablations = A7 / A8; study_pairs builds the
#  user-study images; the study itself needs people: scripts/user_study_app.py; report = every table and figure.)
#
# Speed, with the same data and models: the training images are decoded once into a raw cache (scripts/build_cache.py; set NO_CACHE=1 to
# skip it), and with 2 or more GPUs (e.g. Kaggle "T4 x2") the work runs side by side (with one GPU everything runs one after the other):
#   training   ONE job graph over both GPUs (scripts/train_phase.py): the lambda 0.0130 base (alone, nothing can share it) next to the
#              K=1 ablation base; the 0.0018 fine-tune, then diffusion the moment that codec exists, while the other GPU works through the
#              remaining fine-tunes, the ReLU / K=1 runs, train_ablations, ae and step9. baselines run on the CPU beside them.
#   after      [codec_eval -> bitstream -> ablation_eval -> diffusion_eval -> clic_crops -> protect]  ||
#              [guidance -> honesty -> study_pairs -> intelligence]          then   report
# Things that need another's result are always in one chain or one dependency, so they stay serial: diffusion needs the lambda 0.0018
# codec, protect / guidance need diffusion, honesty / study_pairs need the frozen guidance settings. A stage whose input is missing
# waits (exit 75).
#
# Resume rules: every trainer continues from its own last.pt; a finished stage leaves $EXP/.done/<stage>; a stage that is cut off
# by the time budget (TS_TIME_LIMIT_HOURS) saves and exits with status 75 and is NOT marked done; a crash saves the exact state
# first and is not marked done either. Run the same command again to continue. Logs: $EXP/logs/<stage>.log.
# If TS_SYNC is set (hf:<user>/<repo> or dir:<path>) the run state is uploaded after every finished stage. With TS_CLAIMS=1 every stage is
# also claimed first, so two notebooks on the same repo never run (or overwrite) the same stage: run different STAGES in each.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-cuda}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export TS_SESSION_ID=${TS_SESSION_ID:-$(date +%s)-$$}      # identifies this run in the stage claims (kaggle_run.py sets it)
# shellcheck source=scripts/chains.sh
source scripts/chains.sh
ALL="setup data baselines ae codec codec_eval bitstream step9 diffusion eval_parity diffusion_eval clic_crops protect guidance honesty intelligence codec_ablation ablation_eval train_ablations study_pairs report"
STAGES=${STAGES:-$ALL}

# ---- GPUs: how many chains can run side by side --------------------------------------------------------------------------
if [ -z "${NGPU:-}" ]; then NGPU=$($PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1); fi
if [ "$DEV" = "cpu" ] || [ "${NGPU:-0}" -lt 1 ]; then NGPU=1; fi
export NGPU
WORKERS=$(( NGPU > 1 ? 2 : 4 ))          # data-loader processes per trainer (a Kaggle machine has 4 CPU cores to share)

# ---- scale: guide values by default, tiny values with SMOKE=1 -------------------------------------------------------------
if [ "${SMOKE:-0}" = "1" ]; then
  EXP=experiments_smoke; FIGS=report_smoke/figures; RES=report_smoke/results; STUDY=study_smoke
  AE_IT=300;  AE_BATCH=8;  AE_CROP=128; AMP_IT=100
  C_BASE=400; C_FT=150;    C_BATCH=4;  C_CROP=128; C_N=32; C_M=48
  D_IT=300;   D_BATCH=8;   D_CROP=128; STEPS=10; MAIN_STEPS=10; D_CH="16 32 32 32"
  U_IT=200;   U_BATCH=8;   U_CH="16 32 32 32"; ABL_IT=100; ABL_BATCH=4; ABL_CH="16 32 32 32"
  N_IMG=8;    N_TEXT=8; N_FACE=8; N_TBL=40; N_HALL=12; TUNE_N=4; STUDY_IMG=4; CR_MAX=16; CR_PER=2; CR_BATCH=4; CR_STEPS=4
else
  EXP=experiments; FIGS=report/figures; RES=report/results; STUDY=study
  AE_IT=30000; AE_BATCH=16; AE_CROP=256; AMP_IT=3000
  C_BASE=150000; C_FT=50000; C_BATCH=8; C_CROP=256; C_N=128; C_M=192
  D_IT=200000; D_BATCH=32;  D_CROP=128; STEPS=25; MAIN_STEPS=50; D_CH="64 128 256 256"
  U_IT=20000;  U_BATCH=64;  U_CH="64 128 128 256"; ABL_IT=20000; ABL_BATCH=32; ABL_CH="64 128 256 256"
  N_IMG=500;   N_TEXT=300; N_FACE=500; N_TBL=1500; N_HALL=100; TUNE_N=20; STUDY_IMG=40; CR_MAX=2000; CR_PER=20; CR_BATCH=16; CR_STEPS=$MAIN_STEPS
fi
mkdir -p "$EXP/.done" "$EXP/logs" "$FIGS" "$RES"
# Speed-ups that are used only after a parity check passed (they read these result files; without them everything stays fp32 as in the
# guide): fp16 codec convolutions after `STAGES=amp_parity` (scripts/amp_parity.py), fp16 / batched U-Net decoding after the eval_parity stage.
export TS_AMP_PARITY_FILE=${TS_AMP_PARITY_FILE:-$PWD/$EXP/amp_parity.json}
export TS_EVAL_PARITY_FILE=${TS_EVAL_PARITY_FILE:-$PWD/$EXP/eval_parity.json}
export TS_CODEC_AMP=${TS_CODEC_AMP:-auto} TS_EVAL_AMP=${TS_EVAL_AMP:-auto} TS_EVAL_BATCH_CFG=${TS_EVAL_BATCH_CFG:-auto}
CODEC=$EXP/step7_codec; DIFF=$EXP/step10_diffusion
LO=$CODEC/lmbda0.0018/final.pt; HI=$CODEC/lmbda0.0483/final.pt; UNET=$DIFF/final.pt; FROZEN=$EXP/step12_guidance/frozen.json

nimg() { case "$1" in text) echo $N_TEXT;; face) echo $N_FACE;; *) echo $N_IMG;; esac; }   # images per evaluation set
# claim STAGE: only with TS_CLAIMS=1 (two notebooks working on the same run repo at once, kaggle_run.py --claims) and a storage (TS_SYNC):
# take the stage so no other notebook runs it too. Off by default: a single chain of sessions has nobody to protect against, and the claim
# of a session that was cancelled would otherwise block the stage for 45 minutes.
claim() {
  [ "${TS_CLAIMS:-0}" = "1" ] || return 0
  [ -n "${TS_SYNC:-}" ] || return 0
  local r=0
  $PY -m truesight.sync claim --stage "$1" --dir "$EXP" --spec "$TS_SYNC" || r=$?
  [ "$r" -eq 0 ]
}
want() { case " $STAGES " in *" $1 "*) [ ! -e "$EXP/.done/$1" ] && claim "$1";; *) return 1;; esac; }
run()  { local s=$1; shift; echo "=== [$s] $*"; "$@" 2>&1 | tee -a "$EXP/logs/$s.log"; }   # any failure stops the script (set -e + pipefail)
need() { $PY -m truesight.runtime need "$1" || exit 75; }                                # not enough session time left: pause here
# require FILE STAGE: the file is produced by another stage. Fetch it from the storage if another session made it, else wait.
require() {
  [ -e "$1" ] && return 0
  if [ -n "${TS_SYNC:-}" ]; then $PY -m truesight.sync pull --dir "$EXP" --spec "$TS_SYNC" || true; fi
  [ -e "$1" ] && return 0
  echo "=== $1 does not exist yet: stage '$2' has to finish first (maybe in another notebook). Run this again afterwards."; exit 75
}
done_() {
  touch "$EXP/.done/$1"; echo "=== [$1] finished $(date -u +%FT%TZ)"
  if [ -n "${TS_SYNC:-}" ]; then $PY -m truesight.sync push --dir "$EXP" --spec "$TS_SYNC" || echo "(upload failed, will retry later)"; fi
}

# ============================================================================================ stages
stage_setup() {
  if want setup; then
    $PY - <<'EOF'
import torch; assert torch.cuda.is_available(), "no CUDA GPU visible"; print(torch.cuda.get_device_name(0), torch.version.cuda, "x", torch.cuda.device_count())
EOF
    command -v gcc >/dev/null && command -v ninja >/dev/null || { echo "need gcc + ninja for torchac (apt install build-essential ninja-build)"; exit 1; }
    run setup $PY -c "import torchac; print('torchac ok')"          # compiles the extension once
    # The tests run in a clean environment: without this session's TS_* variables and without the notebook's secrets, so they can neither be
    # confused by them nor touch the real run state. A failing test is reported but does not stop the run (SETUP_TESTS_STRICT=1 makes it
    # fatal): the checks that must hold (GPU visible, torchac compiled) are made above and in the line before.
    if ! run setup env -u TS_TIME_LIMIT_HOURS -u TS_SESSION_START -u TS_SESSION_ID -u TS_SYNC -u TS_CLAIMS -u TS_DATA_REPO -u TS_SAVE_EVERY_MIN \
         -u TS_CODEC_AMP -u TS_EVAL_AMP -u TS_EVAL_BATCH_CFG -u TS_AMP_PARITY_FILE -u TS_EVAL_PARITY_FILE -u TS_CLAIM_TTL_MIN \
         TS_NO_KAGGLE_SECRETS=1 $PY -m pytest tests -q -p no:cacheprovider; then
      echo "=== WARNING: some tests failed (see above). Continuing: set SETUP_TESTS_STRICT=1 to stop here instead."
      [ "${SETUP_TESTS_STRICT:-0}" = "1" ] && exit 1
    fi
    done_ setup
  fi
}

stage_data() {
  if want data; then                                              # ~23 GB download (skipped when restored), then filter / dedupe / split
    need 30
    run data $PY scripts/prepare_data.py --stage all --workers 8
    wc -l data/processed/*.txt | tee -a "$EXP/logs/data.log"
    done_ data
  fi
}

stage_baselines() {
  if want baselines; then                                         # Steps 3-5 (CPU work)
    need 20
    run baselines $PY scripts/eval_rd.py --sets kodak clic
    run baselines $PY scripts/eval_course_baselines.py
    run baselines $PY scripts/train_size_predictor.py
    done_ baselines
  fi
}

stage_ae() {
  if want ae; then                                                # Step 6: PlainAE, M = 8, 16, 32
    run ae $PY scripts/train_ae.py --iters $AE_IT --batch $AE_BATCH --crop $AE_CROP --workers $WORKERS --out "$EXP/step6_ae" --figs "$FIGS"
    done_ ae
  fi
}

stage_codec() {
  if want codec; then                                             # Step 7: lambda 0.0130 full run, then five fine-tunes (no ablations yet)
    # train_all_lambdas trains the base alone, then the fine-tunes one per visible GPU (it only needs the base, not each other)
    run codec $PY scripts/train_all_lambdas.py --out "$CODEC" --base-iters $C_BASE --finetune-iters $C_FT \
        --batch $C_BATCH --crop $C_CROP --N $C_N --M $C_M --workers $WORKERS --no-ablation
    done_ codec
  fi
}

stage_codec_eval() {
  if want codec_eval; then                                        # RD curves, BD-rate vs JPEG / WebP, LPIPS, MS-SSIM on Kodak
    need 20
    require "$LO" codec
    run codec_eval $PY scripts/eval_codec.py --exp "$CODEC" --figs "$FIGS"
    done_ codec_eval
  fi
}

stage_bitstream() {
  if want bitstream; then                                         # Step 8: real .sphota files, bit-exact decode, header counted
    need 20
    require "$LO" codec
    run bitstream $PY scripts/eval_bitstream.py --exp "$CODEC" --out "$EXP/step8_bitstream" --figs "$FIGS" --device cpu
    done_ bitstream
  fi
}

stage_step9() {
  if want step9; then                                             # Step 9 warm-up: unconditional 64x64 DDPM + forward-process figure
    run step9 $PY scripts/step9_ddpm.py --out "$EXP/step9_ddpm" --figs "$FIGS" --iters $U_IT --batch $U_BATCH --channels $U_CH --workers $WORKERS
    done_ step9
  fi
}

stage_diffusion() {
  if want diffusion; then                                         # Step 10: conditional decoder on the lowest-lambda codec
    require "$LO" codec
    run diffusion $PY -m truesight.diffusion.train_diffusion --codec "$LO" --out "$DIFF" \
        --iters $D_IT --batch $D_BATCH --crop $D_CROP --channels $D_CH --workers $WORKERS
    done_ diffusion
  fi
}

stage_eval_parity() {
  if want eval_parity; then                                       # does fp16 / batched U-Net decoding give the same decodes? (5-10 min)
    need 15
    require "$UNET" diffusion
    run eval_parity $PY scripts/eval_parity.py --codec "$LO" --diffusion "$UNET" --out "$EXP/eval_parity.json" --device $DEV
    done_ eval_parity
  fi
}

stage_amp_parity() {
  if want amp_parity; then                                        # fp16 codec convolutions vs fp32: short A/B (not in the default run)
    need 90
    run amp_parity $PY scripts/amp_parity.py --out "$EXP/amp_parity" --iters $AMP_IT --batch $C_BATCH --crop $C_CROP --N $C_N --M $C_M         --workers $WORKERS --device $DEV
    done_ amp_parity
  fi
}

stage_diffusion_eval() {
  if want diffusion_eval; then                                    # CFG sweep (A3), steps sweep (A6), LPIPS / PSNR / FID vs the codec decode
    need 45
    require "$UNET" diffusion
    run diffusion_eval $PY scripts/eval_diffusion.py --codec "$LO" --diffusion "$UNET" --out "$DIFF/eval" --figs "$FIGS" \
        --steps $STEPS --main-steps $MAIN_STEPS --device $DEV --fid
    done_ diffusion_eval
  fi
}

stage_clic_crops() {
  if want clic_crops; then                                        # LPIPS + FID on ~2,000 matched 256x256 CLIC crops (resumable, keeps its batches)
    need 60
    require "$UNET" diffusion
    run clic_crops $PY scripts/eval_clic_crops.py --codec "$LO" --diffusion "$UNET" --out "$DIFF/clic_crops" --steps $CR_STEPS \
        --crops-per-image $CR_PER --max-crops $CR_MAX --batch $CR_BATCH --fid --device $DEV
    done_ clic_crops
  fi
}

stage_protect() {
  if want protect; then                                           # Step 11: ROI coding, TFR / FIR on the text and face sets
    require "$UNET" diffusion
    require "$HI" codec
    for S in text face; do
      need 30
      run protect $PY scripts/eval_protect.py --set $S --limit $(nimg $S) --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" \
          --steps $STEPS --tfr --fir --device $DEV --figs "$FIGS" --out "$EXP/step11_protect/$S"
    done
    done_ protect
  fi
}

stage_guidance() {
  if want guidance; then                                          # Step 12: tune on val once, freeze, evaluate once on the test sets (A4, A5)
    need 60
    require "$UNET" diffusion
    require "$HI" codec
    run guidance $PY scripts/tune_guidance.py --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" --limit $TUNE_N \
        --steps $MAIN_STEPS --tfr --fir --device $DEV --out "$EXP/step12_guidance"
    for S in text face kodak clic; do
      need 30
      run guidance $PY scripts/eval_sphota.py --set $S --limit $(nimg $S) --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" \
          --frozen "$FROZEN" --steps $MAIN_STEPS --tfr --fir --device $DEV --figs "$FIGS" --out "$EXP/step12_guidance/eval_$S"
    done
    done_ guidance
  fi
}

stage_honesty() {
  if want honesty; then                                           # Step 13: per-patch signals, 8 detectors, learning curves, safe mode
    need 60
    require "$FROZEN" guidance
    run honesty $PY scripts/eval_hallucination.py --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" --limit $N_HALL \
        --steps $STEPS --guided --frozen "$FROZEN" --with-roi --encoder-boxes --tfr \
        --device $DEV --figs "$FIGS" --out "$EXP/step13_honesty"
    done_ honesty
  fi
}

stage_intelligence() {
  if want intelligence; then                                      # Step 14: real byte counts, generative modes, OCR + face checks
    need 60
    require "$UNET" diffusion
    require "$HI" codec
    run intelligence $PY scripts/build_mode_table.py --list data/processed/train.txt --limit $N_TBL --codecs-dir "$CODEC" \
        --real-bytes --diffusion "$UNET" --steps $STEPS --ocr --faces --device $DEV --out "$EXP/step14_intelligence"
    run intelligence $PY scripts/train_intelligence.py --data "$EXP/step14_intelligence" --figs "$FIGS"
    done_ intelligence
  fi
}

stage_codec_ablation() {
  if want codec_ablation; then                                    # A1 (K = 1) and A2 (ReLU instead of GDN) at lambda 0.0130 and 0.0035
    # on one GPU the K=1 and ReLU chains run one after the other; given two GPUs they train side by side
    run codec_ablation $PY scripts/train_all_lambdas.py --out "$CODEC" --base-iters $C_BASE --finetune-iters $C_FT \
        --batch $C_BATCH --crop $C_CROP --N $C_N --M $C_M --workers $WORKERS --only-ablation
    done_ codec_ablation
  fi
}

stage_ablation_eval() {
  if want ablation_eval; then                                     # the same evaluation again: now it also writes ablation_k.csv and ablation_act.csv
    need 20
    require "$CODEC/K1_lmbda0.0130/final.pt" codec_ablation
    run ablation_eval $PY scripts/eval_codec.py --exp "$CODEC" --figs "$FIGS"
    done_ ablation_eval
  fi
}

stage_train_ablations() {
  if want train_ablations; then                                   # A7 (linear vs cosine schedule) and A8 (Adam vs AdamW, constant vs cosine LR)
    require "$LO" codec
    run train_ablations $PY scripts/run_train_ablations.py --codec "$LO" --out "$EXP/step10_ablation" --iters $ABL_IT \
        --batch $ABL_BATCH --channels $ABL_CH --workers $WORKERS --device $DEV --steps $STEPS
    done_ train_ablations
  fi
}

stage_study_pairs() {
  if want study_pairs; then                                       # images for the user study (the answers need real people)
    need 30
    require "$FROZEN" guidance
    run study_pairs $PY scripts/make_study_pairs.py --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" --frozen "$FROZEN" \
        --images $STUDY_IMG --out "$STUDY" --device $DEV
    done_ study_pairs
  fi
}

stage_report() {
  if want report; then                                            # every table and figure from the evaluation outputs
    run report $PY scripts/eval_faithfulness.py --exp "$EXP" --out "$RES" --figs "$FIGS"
    run report $PY scripts/make_figures.py --exp "$EXP" --out "$RES" --figs "$FIGS" --study "$STUDY"
    run report $PY scripts/build_report.py --results "$RES" --out "$RES/../report.md"
    done_ report
  fi
}

# ============================================================================================ cache, job graph, chains
ensure_cache() {                      # decode the training images once (truesight/image_cache.py): same data, no PNG decoding per step
  [ -z "${TS_INNER:-}" ] || return 0
  [ "${SMOKE:-0}" = "1" ] || [ "${NO_CACHE:-0}" = "1" ] && return 0
  case " $STAGES " in *" ae "*|*" step9 "*|*" codec "*|*" codec_ablation "*|*" diffusion "*|*" train_ablations "*) ;; *) return 0;; esac
  [ -s data/processed/train.txt ] || return 0
  $PY scripts/build_cache.py --list data/processed/train.txt --out "${TS_IMAGE_CACHE:-data/cache}" --workers 4     || echo "(image cache failed: training decodes the PNG files instead, same data, slower)"
}

# phase_train: with 2+ GPUs run the heavy training stages as ONE job graph (scripts/train_phase.py): a GPU never waits while something
# is ready (the K=1 / ReLU bases train beside the 0.0130 base, diffusion starts when the 0.0018 codec exists). Returns 1 when it does
# not apply (one GPU, a single wanted stage, or we are already inside a job of the graph): the serial chains below run instead.
phase_train() {
  [ -z "${TS_INNER:-}" ] && [ "$NGPU" -ge 2 ] || return 1
  local list="" s bpid="" rc=0 b=0
  for s in ae step9 codec codec_ablation diffusion train_ablations; do want "$s" && list="$list $s"; done
  set -- $list
  [ $# -ge 2 ] || return 1
  stage_baselines & bpid=$!                                      # CPU-only work, beside the GPUs
  $PY scripts/train_phase.py --exp "$EXP" --codec-dir "$CODEC" --stages "$list" --base-iters $C_BASE --finetune-iters $C_FT       --batch $C_BATCH --crop $C_CROP --N $C_N --M $C_M --workers $WORKERS || rc=$?
  wait "$bpid" || b=$?
  [ "$rc" -eq 0 ] || exit "$rc"
  [ "$b" -eq 0 ] || exit "$b"
  return 0
}

chain_cpu_then_ae()      { stage_baselines; stage_ae; }
chain_step9()            { stage_step9; }
chain_diffusion()        { stage_diffusion; }
chain_train_side()       { stage_codec_ablation; stage_train_ablations; }
chain_post_a()           { stage_codec_eval; stage_bitstream; stage_ablation_eval; stage_diffusion_eval; stage_clic_crops; stage_protect; }
chain_post_b()           { stage_guidance; stage_honesty; stage_study_pairs; stage_intelligence; }

stage_setup
stage_data
ensure_cache
stage_amp_parity                                                 # only when requested (STAGES=amp_parity); decides TS_CODEC_AMP=auto
if phase_train; then
  stage_codec; stage_codec_ablation                              # their runs are finished: this only writes the done-markers
else
  run_chains chain_cpu_then_ae chain_step9
  stage_codec
  run_chains chain_diffusion chain_train_side
fi
stage_eval_parity                                                # decides TS_EVAL_AMP / TS_EVAL_BATCH_CFG = auto for the evaluation
run_chains chain_post_a chain_post_b
stage_report

echo "ALL REQUESTED STAGES DONE. Results: $EXP/ (tables, checkpoints), $RES/ and $FIGS/"
