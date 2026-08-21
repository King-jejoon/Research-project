#!/bin/zsh
# v6 transfer chain runner: DS01 stage-2 re-freeze -> wait DS02 -> DS08a -> DS07
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
export NTHREADS=12
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "DS01 stage2 re-run (tie-band rule, cached GPs)"
DS=ds01 $PY exp_dsx_stage2.py > /dev/null 2>&1 && log "ds01 stage2 done" || log "ds01 stage2 FAILED"
log "waiting for the DS02 stage2 process to finish"
while pgrep -f "exp_dsx_stage2.py" > /dev/null; do sleep 30; done
log "DS02 finished; starting DS08a / DS07 chains"
for ds in ds08a ds07; do
  for step in c3 detsel hi1 stage2; do
    log "$ds $step start"
    DS=$ds $PY exp_dsx_$step.py > ${ds}v6_${step}_run.log 2>&1 && log "$ds $step done" || { log "$ds $step FAILED (see ${ds}v6_${step}_run.log)"; break; }
  done
done
log "chain finished"
