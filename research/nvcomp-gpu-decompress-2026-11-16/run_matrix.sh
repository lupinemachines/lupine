#!/bin/bash
# Runs the phase-1 microbenchmark matrix on node-008, each run under the shared bench lock.
# Skips configurations already present in the output file.
out=${1:?out file}
reps=${REPS:-3}
run() { # file dev codec level chunk threads
  local gpu
  gpu=$([ "$2" = 0 ] && echo "RTX 4090" || echo "2070 SUPER")
  grep -q "$gpu\" file=$1 codec=$3 level=$4 chunk_kib=$5 threads=$6 path=" "$out" 2>/dev/null && return
  ssh kevin@inferable-node-006 "timeout 600 flock /tmp/lupine-agent-bench.lock ssh -o BatchMode=yes kevin@inferable-node-008 'cd ~/lupine-agents/x1-nvcomp && timeout 300 bench/gpudec_bench $* $reps'" >>"$out" 2>&1
}
for t in 1 4; do
  for d in 0 1; do
    run data/zeros.bin $d none 0 64 $t
    for f in zeros tokens_i64 text f32_sparse90 bf16_weights f32_normal random; do
      for c in 64 256 1024; do
        run data/$f.bin $d zstd -1 $c $t
        run data/$f.bin $d lz4 0 $c $t
      done
      for c in 64 256; do run data/$f.bin $d zstd 1 $c $t; done
      run data/$f.bin $d gdeflate 1 64 $t
    done
  done
done
echo MATRIX_DONE >>"$out"
