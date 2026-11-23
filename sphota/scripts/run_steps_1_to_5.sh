#!/usr/bin/env bash
# Re-run Steps 1-5 from the project root and check them end to end. Usage: bash scripts/run_steps_1_to_5.sh
set -e
PY=${PY:-python}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
echo "=== unit tests (all steps)";            $PY -m pytest tests -q
echo "=== Step 1: download (no-op if present) + prepare"; $PY scripts/prepare_data.py --stage all
echo "=== Step 3: JPEG/WebP RD on Kodak";     $PY scripts/eval_rd.py --sets kodak
echo "=== Step 4: PCA + VQ baselines";        $PY scripts/eval_course_baselines.py
echo "=== Step 5: features + size predictor"; $PY scripts/train_size_predictor.py
echo "=== end-to-end checks (Steps 1-5)";     $PY scripts/e2e_check.py
