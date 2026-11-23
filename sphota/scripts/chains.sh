#!/usr/bin/env bash
# run_chains: run shell functions ("chains") side by side, one per GPU, or one after the other when there is only one GPU.
# Sourced by scripts/run_full_pipeline.sh (and by tests/test_chains.py).
#
#   NGPU=2 run_chains chain_a chain_b      # chain_a on GPU 0, chain_b on GPU 1, at the same time
#
# Inside a chain the steps still run strictly one after the other. Put steps that need each other's result in ONE chain, and only
# steps that are independent in different chains. Needs set -e in the caller (a failing step stops its chain).
#
# Outcome: returns 0 when every chain finished, 75 when one paused (time budget; the others pause themselves at the same budget),
# otherwise the status of the first chain that crashed. When one chain fails the others get SIGTERM so each trainer saves its exact
# state before stopping (a SIGTERM to a trainer is "save and stop", see truesight/runtime.py). Each chain writes its own log
# ($EXP/logs/<stage>.log) and its own folders, so chains never write the same file.

run_chains() {
  local c
  if [ "${NGPU:-1}" -lt 2 ] || [ $# -lt 2 ]; then       # one GPU: serial, in the given order (errexit ends the script on a failure)
    for c in "$@"; do "$c"; done
    return 0
  fi
  set -m                                                # every chain gets its own process group, so a stop reaches its python children
  local -a pids=()
  local i=0 alive=0 rc=0 stopping=0 r k p q
  for c in "$@"; do
    ( export CUDA_VISIBLE_DEVICES=$((i % NGPU)); "$c" ) &
    pids+=("$!"); i=$((i + 1)); alive=$((alive + 1))
  done
  while [ "$alive" -gt 0 ]; do
    for k in "${!pids[@]}"; do
      p=${pids[$k]}
      [ -n "$p" ] || continue
      if ! kill -0 "$p" 2>/dev/null; then
        r=0; wait "$p" || r=$?
        pids[$k]=""; alive=$((alive - 1))
        if [ "$r" -ne 0 ]; then
          if [ "$rc" -eq 0 ] || { [ "$rc" -eq 75 ] && [ "$r" -ne 75 ]; }; then rc=$r; fi     # a crash outranks a pause
          if [ "$stopping" -eq 0 ]; then
            stopping=1
            for q in "${pids[@]}"; do
              [ -n "$q" ] || continue
              kill -TERM -- "-$q" 2>/dev/null || kill -TERM "$q" 2>/dev/null || true
            done
          fi
        fi
      fi
    done
    [ "$alive" -gt 0 ] && sleep "${CHAIN_POLL:-2}"
  done
  set +m
  return "$rc"
}
