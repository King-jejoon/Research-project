#!/bin/zsh
# v3: fully seed-parallel; DS08a and DS07 branches run concurrently
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
log() { echo "[$(date '+%H:%M:%S')] $*"; }
waitfor() { while [ ! -f "$1" ]; do sleep 20; done; }
par3() { # $1=ds $2=script $3=extra env
  for s in 0 1 2; do env SEEDS_ONLY=$s $3 NTHREADS=4 DS=$1 $PY $2 > ${1}v6_$(basename $2 .py)_s${s}_run.log 2>&1 & done; wait
}
branch() {
  ds=$1
  if [ "$ds" = "ds08a" ]; then
    waitfor ds08av6_c3_stats_s0.npz; log "$ds c3 seeds ready"
    NTHREADS=4 DS=$ds $PY exp_dsx_c3.py > ${ds}v6_c3_run.log 2>&1 || { log "$ds c3 merge FAILED"; return; }
    NTHREADS=4 DS=$ds $PY exp_dsx_detsel.py > ${ds}v6_detsel_run.log 2>&1 || { log "$ds detsel FAILED"; return; }; log "$ds detsel done"
    NTHREADS=4 DS=$ds $PY exp_dsx_hi1.py > ${ds}v6_hi1_run.log 2>&1 || { log "$ds hi1 FAILED"; return; }; log "$ds hi1 done"
    par3 $ds exp_dsx_stage2.py "STATS_ONLY=1"; log "$ds stage2 seeds ready"
  else
    waitfor ds07v6_hr_stats_s0.npz; waitfor ds07v6_hr_stats_s1.npz; waitfor ds07v6_hr_stats_s2.npz; log "$ds stage2 seeds ready (prebuilt)"
  fi
  NTHREADS=4 DS=$ds $PY exp_dsx_stage2.py > ${ds}v6_stage2_run.log 2>&1 || { log "$ds stage2 merge FAILED"; return; }; log "$ds stage2 done"
  par3 $ds exp_dsx_test.py "STATS_ONLY=1"; log "$ds test prebuild done"
  NTHREADS=4 DS=$ds $PY exp_dsx_test.py > ${ds}v6_test_run.log 2>&1 && log "$ds TEST DONE" || log "$ds test FAILED"
}
branch ds08a &
branch ds07 &
wait
log "v3 chain finished"
