#!/bin/bash
# One bench-lock hold per (round, dataset), all GPUs and configs inside it.
S=$(dirname "$0")
out=$1
for r in 1 2 3 4 5; do
  for f in zeros tokens_i64 text f32_sparse90 bf16_weights random; do
    reps=6; case $f in bf16_weights|random) reps=4 ;; esac
    if grep -q "^round=$r gpu=4090 cfg=lz4-64 reps=[1-9].*file=data/$f.bin" "$out" 2>/dev/null; then continue; fi
    cp "$out" "$out.tmp" 2>/dev/null || : > "$out.tmp"
    n=$(wc -l < "$out.tmp")
    "$S/lockrun.sh" env NOLOCK=1 ROUND_FROM=$r ROUNDS=$r DATASETS=$f REPS=$reps CONFIGS="off zstd64 lz4-64" "$S/e2e2.sh" "$out.tmp"
    tail -n +$((n + 1)) "$out.tmp" | grep -v E2E2_DONE >> "$out"; rm -f "$out.tmp"
  done
done
echo E2E2_DONE >> "$out"
