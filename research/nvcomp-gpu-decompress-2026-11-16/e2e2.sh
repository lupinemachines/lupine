#!/bin/bash
# Driven from the workstation: client on node-006 (under the shared bench lock),
# server on node-008. Server CPU = listener's reaped-children CPU delta.
out=${1:?out}
rounds=${ROUNDS:-5}
datasets=${DATASETS:-"zeros tokens_i64 text f32_sparse90 bf16_weights random"}
configs=${CONFIGS:-"off zstd64 zstd256 lz4-64 zstd64-cpu"}
reps=${REPS:-3}
srvcpu() {
  ssh kevin@inferable-node-008 "p=\$(cat ~/lupine-agents/x1-nvcomp/server-$1.pid); for i in \$(seq 200); do pgrep -P \$p >/dev/null || break; sleep 0.05; done; awk '{print \$14+\$15+\$16+\$17}' /proc/\$p/stat"
}
run() { # round data gpu cfg reps
  local port env
  case $3/$4 in
    4090/zstd64-cpu) port=16253 ;; 2070/zstd64-cpu) port=16254 ;;
    4090/*) port=16251 ;; 2070/*) port=16252 ;;
  esac
  case $4 in
    off) env="" ;;
    zstd64|zstd64-cpu) env="LUPINE_NVCOMP_HTOD=64" ;;
    zstd256) env="LUPINE_NVCOMP_HTOD=256" ;;
    lz4-64) env="LUPINE_NVCOMP_HTOD=64 LUPINE_NVCOMP_HTOD_CODEC=lz4" ;;
  esac
  local c0 c1 line
  grep -q "^round=$1 gpu=$3 cfg=$4 reps=$5 .*file=data/$2.bin" "$out" 2>/dev/null && return
  c0=$(srvcpu $port)
  line=$(ssh kevin@inferable-node-006 "cd ~/lupine-agents/x1-nvcomp && timeout 300 $lock env $env CUDA_VISIBLE_DEVICES=1 LUPINE_SERVER=192.168.1.131:$port LD_LIBRARY_PATH=src/build ./htod_e2e data/$2.bin $5 2>&1 | tail -1")
  c1=$(srvcpu $port)
  echo "round=$1 gpu=$3 cfg=$4 reps=$5 server_ticks=$((c1 - c0)) $line" >>"$out"
}
lock="flock /tmp/lupine-agent-bench.lock"
[ -n "$NOLOCK" ] && lock=""
for r in $(seq "${ROUND_FROM:-1}" "$rounds"); do
  for f in $datasets; do
    for g in 2070 4090; do
      for c in $configs; do
        [ "$r" = 1 ] && run "$r" "$f" "$g" "$c" 0
        run "$r" "$f" "$g" "$c" "$reps"
      done
    done
  done
done
echo E2E2_DONE >>"$out"
