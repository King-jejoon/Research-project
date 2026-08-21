#!/bin/zsh
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
export NTHREADS=12
log() { echo "[$(date '+%H:%M:%S')] $*"; }
for ds in ds08a ds07; do
  for step in c3 detsel hi1 stage2; do
    log "$ds $step start"
    DS=$ds $PY exp_dsx_$step.py > ${ds}v6_${step}_run.log 2>&1 && log "$ds $step done" || { log "$ds $step FAILED (see ${ds}v6_${step}_run.log)"; break; }
  done
done
log "chain finished"
