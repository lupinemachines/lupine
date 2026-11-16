# 16251 4090 nvCOMP, 16252 2070 nvCOMP, 16253 4090 no nvCOMP, 16254 2070 no nvCOMP
cd ~/lupine-agents/x1-nvcomp/src/build
NV=$HOME/lupine-regress/libsamples/nvcomp/lib
U4090=GPU-14ab7d48-3479-e25f-fdac-25ca1ce330ff
U2070=GPU-74bdd3d3-3f40-9c98-0019-a36a16f7243d
start() { setsid bash -c "exec env $1 LUPINE_PORT=$2 ./lupine_driver_server" </dev/null >~/lupine-agents/x1-nvcomp/server-$2.log 2>&1 & echo $! > ~/lupine-agents/x1-nvcomp/server-$2.pid; }
start "CUDA_VISIBLE_DEVICES=$U4090 LUPINE_NVCOMP_HTOD=1 LUPINE_NVCOMP_HTOD_STATS=1 LD_LIBRARY_PATH=$NV" 16251
start "CUDA_VISIBLE_DEVICES=$U2070 LUPINE_NVCOMP_HTOD=1 LUPINE_NVCOMP_HTOD_STATS=1 LD_LIBRARY_PATH=$NV" 16252
start "CUDA_VISIBLE_DEVICES=$U4090" 16253
start "CUDA_VISIBLE_DEVICES=$U2070" 16254
sleep 1
