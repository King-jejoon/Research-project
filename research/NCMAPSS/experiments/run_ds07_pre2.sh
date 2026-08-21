#!/bin/zsh
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
log() { echo "[$(date '+%H:%M:%S')] $*"; }
NTHREADS=4 DS=ds07 $PY exp_dsx_c3.py > ds07v6_c3_run.log 2>&1 && log "ds07 c3 merge done" || { log "ds07 c3 merge FAILED"; exit 1; }
NTHREADS=4 DS=ds07 $PY exp_dsx_detsel.py > ds07v6_detsel_run.log 2>&1 && log "ds07 detsel done" || { log "ds07 detsel FAILED"; exit 1; }
NTHREADS=4 DS=ds07 $PY exp_dsx_hi1.py > ds07v6_hi1_run.log 2>&1 && log "ds07 hi1 done" || { log "ds07 hi1 FAILED"; exit 1; }
for s in 0 1 2; do SEEDS_ONLY=$s STATS_ONLY=1 NTHREADS=3 DS=ds07 $PY exp_dsx_stage2.py > ds07v6_stage2_s${s}_run.log 2>&1 & done; wait
log "ds07 stage2 stats prebuilt"
