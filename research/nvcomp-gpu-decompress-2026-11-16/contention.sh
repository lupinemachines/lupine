#!/bin/bash
# Runs on node-006 under the bench lock: upload alone, then a second connection's
# FMA workload alone, then three uploads while it runs. contention.sh <port> <data> <cfg>
cd ~/lupine-agents/x1-nvcomp
port=$1 data=$2 cfg=$3
case $cfg in
  off) env="" ;;
  zstd64) env="LUPINE_NVCOMP_HTOD=64" ;;
  lz4-64) env="LUPINE_NVCOMP_HTOD=64 LUPINE_NVCOMP_HTOD_CODEC=lz4" ;;
esac
base="CUDA_VISIBLE_DEVICES=1 LUPINE_SERVER=192.168.1.131:$port LD_LIBRARY_PATH=src/build"
echo "ALONE $(env $env $base ./htod_e2e data/$data.bin 4 | tail -1)"
env $base ./gpuload burn.ptx 45 >load.$$ 2>&1 &
sleep 12
for i in 1 2 3; do
  echo "UPSTART $(date +%s.%N)"
  echo "LOADED $(env $env $base ./htod_e2e data/$data.bin 4 | tail -1)"
  echo "UPEND $(date +%s.%N)"
  sleep 2
done
wait
cat load.$$
rm -f load.$$
