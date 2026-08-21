#!/bin/zsh
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
log() { echo "[$(date '+%H:%M:%S')] $*"; }
# DS02: stage-2 downstream for seeds 0,1 (stats cached), then test on 0,1
( SEEDS_ONLY=0,1 NTHREADS=2 DS=ds02 $PY exp_dsx_stage2.py > ds02v6_stage2_s01_run.log 2>&1 && log "ds02 stage2(0,1) done" && SEEDS_ONLY=0,1 NTHREADS=2 DS=ds02 $PY exp_dsx_test.py > ds02v6_test_run.log 2>&1 && log "ds02 TEST DONE" || log "ds02 FAILED" ) &
( NTHREADS=2 DS=ds01 $PY exp_dsx_test.py > ds01v6_test_run.log 2>&1 && log "ds01 TEST DONE" || log "ds01 FAILED" ) &
for ds in ds04 ds05 ds06 ds08c; do
  ( SEEDS_ONLY=0 NTHREADS=2 DS=$ds $PY exp_dsx_test.py > ${ds}v6_test_run.log 2>&1 && log "$ds TEST DONE" || log "$ds FAILED" ) &
done
wait; log "test screen finished"
