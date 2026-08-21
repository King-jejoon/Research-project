#!/bin/zsh
# single-seed screen of DS04/DS05/DS06/DS08c: gate effect (wide axis) + stage-2 RUL effect on dev
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
log() { echo "[$(date '+%H:%M:%S')] $*"; }
DSS=(ds04 ds05 ds06 ds08c)
for ds in $DSS; do [ -f cache_$ds.npz ] || DS=$ds $PY build_cache_dsx.py > /dev/null 2>&1; done; log "caches ready"
for ds in $DSS; do SEEDS_ONLY=0 NTHREADS=3 DS=$ds $PY exp_dsx_c3.py > ${ds}v6_c3_s0_run.log 2>&1 & done; wait; log "c3 seed0 done (4 datasets)"
for ds in $DSS; do
  SEEDS_ONLY=0 NTHREADS=3 DS=$ds $PY exp_dsx_detsel.py > ${ds}v6_detsel_run.log 2>&1 && log "$ds detsel done" || log "$ds detsel FAILED"
done
for ds in $DSS; do SEEDS_ONLY=0 NTHREADS=3 DS=$ds $PY exp_dsx_hi1.py > ${ds}v6_hi1_run.log 2>&1 & done; wait; log "hi1 done"
SEEDS_ONLY=0 DSLIST=ds04,ds05,ds06,ds08c DIAG_OUT=dsx_screen_gate_results.txt $PY exp_dsx_gate_diag.py > dsx_screen_gate_run.log 2>&1 && log "gate diag done"
for ds in $DSS; do SEEDS_ONLY=0 STATS_ONLY=1 NTHREADS=3 DS=$ds $PY exp_dsx_stage2.py > ${ds}v6_stage2_s0_run.log 2>&1 & done; wait; log "stage2 seed0 fits done"
for ds in $DSS; do SEEDS_ONLY=0 NTHREADS=3 DS=$ds $PY exp_dsx_stage2.py > ${ds}v6_stage2_run.log 2>&1 && log "$ds stage2 done" || log "$ds stage2 FAILED"; done
log "screen finished"
