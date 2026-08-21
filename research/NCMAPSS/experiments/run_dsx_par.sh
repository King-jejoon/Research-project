#!/bin/zsh
# seed-parallel v6 chain: c3 (3 procs) -> merge -> detsel -> hi1 -> stage2 stats (3 procs) -> merge+downstream -> test
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
log() { echo "[$(date '+%H:%M:%S')] $*"; }
for ds in ds08a ds07; do
  log "$ds c3: 3 seed processes"
  for s in 0 1 2; do SEEDS_ONLY=$s NTHREADS=4 DS=$ds $PY exp_dsx_c3.py > ${ds}v6_c3_s${s}_run.log 2>&1 & done; wait
  NTHREADS=12 DS=$ds $PY exp_dsx_c3.py > ${ds}v6_c3_run.log 2>&1 || { log "$ds c3 merge FAILED"; continue; }
  log "$ds c3 done"
  NTHREADS=12 DS=$ds $PY exp_dsx_detsel.py > ${ds}v6_detsel_run.log 2>&1 || { log "$ds detsel FAILED"; continue; }
  log "$ds detsel done"
  NTHREADS=12 DS=$ds $PY exp_dsx_hi1.py > ${ds}v6_hi1_run.log 2>&1 || { log "$ds hi1 FAILED"; continue; }
  log "$ds hi1 done"
  log "$ds stage2 stats: 3 seed processes"
  for s in 0 1 2; do SEEDS_ONLY=$s STATS_ONLY=1 NTHREADS=4 DS=$ds $PY exp_dsx_stage2.py > ${ds}v6_stage2_s${s}_run.log 2>&1 & done; wait
  NTHREADS=12 DS=$ds $PY exp_dsx_stage2.py > ${ds}v6_stage2_run.log 2>&1 || { log "$ds stage2 merge FAILED"; continue; }
  log "$ds stage2 done"
  NTHREADS=12 DS=$ds $PY exp_dsx_test.py > ${ds}v6_test_run.log 2>&1 && log "$ds test done" || log "$ds test FAILED"
done
log "chain finished"
