#!/bin/zsh
cd /Users/a1/Desktop/Research-project/research/NCMAPSS/experiments
PY=/opt/anaconda3/envs/pt_prac/bin/python3
for s in 0 1 2; do SEEDS_ONLY=$s NTHREADS=3 DS=ds07 $PY exp_dsx_c3.py > ds07v6_c3_s${s}_run.log 2>&1 & done; wait
echo "[$(date '+%H:%M:%S')] ds07 c3 seeds prebuilt"
