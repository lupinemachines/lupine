#!/bin/bash
# Runs on node-006. 5 interleaved rounds: opt-in off vs framed (64/256 KiB) against the
# nvCOMP server, plus framed-64 against the CPU-decoding server (no nvCOMP).
cd ~/lupine-agents/x1-nvcomp
out=${1:-e2e.txt}
B=src/build
rounds=${ROUNDS:-5}
datasets=${DATASETS:-"zeros tokens_i64 text f32_sparse90 bf16_weights f32_normal random"}
for r in $(seq 1 "$rounds"); do
  for f in $datasets; do
    for gpu in 4090 2070; do
      if [ $gpu = 4090 ]; then on=16251; off=16253; else on=16252; off=16254; fi
      for cfg in "off $on" "64 $on" "256 $on" "64 $off"; do
        set -- $cfg
        timeout 300 flock /tmp/lupine-agent-bench.lock env \
          $([ "$1" != off ] && echo LUPINE_NVCOMP_HTOD=$1) \
          CUDA_VISIBLE_DEVICES=1 LUPINE_SERVER=192.168.1.131:$2 LD_LIBRARY_PATH=$B \
          ./htod_e2e data/$f.bin 3 2>&1 | sed "s/^/round=$r gpu=$gpu port=$2 /" >>"$out"
      done
    done
  done
done
echo E2E_DONE >>"$out"
