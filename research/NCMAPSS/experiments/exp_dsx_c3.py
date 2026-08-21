"""
exp_dsx_c3.py — DS01/DS02 transfer, PHASE 1a: stage-1 sparse-window MOGP
fits + dev point statistics.  Dev only, no test contact.

Transfer principle (user decisions 2026-08-19): model classes, pipeline
structure, loss-term TYPES and every selection PROCEDURE are carried over
from DS03; every numeric constant (detector window, clip, gate V, HI
lambda1/lambda2, beta, z-norm) is RE-SELECTED on the new dataset's dev
split with the identical DS03 procedure.  Table-3 recipe constants
(lambda0, thr, end_target, flat w/m) stay structural (user decision).

This script only builds the raw material for the re-selection:
  per seed (0,1,2): cycle<=3 stratified train rows (1000 per unit-cycle;
  6 units x 3 cycles = 18k — the DS03 budget RULE, dataset-scaled) ->
  Vecchia MOGP fit (frozen internals: rbf rank 1, m=18, 5 sensors) ->
  dev point stats on the chain grid (NPER=200, rng u*7+sd) with per-unit
  cc/resid/detcov/ll + ucyc/hours/onset.
Hours = per-cycle row count / 3600 (cache verified 1 Hz full-resolution
against DS03 rul_input).  Onset label = first cycle with mean hs < 0.5
(chain convention).
Usage: DS=ds01|ds02 python3 exp_dsx_c3.py
Outputs: {DS}v6_c3_results.txt, {DS}v6_gp_c3_s{sd}.pt, {DS}v6_c3_stats_s{sd}.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
from exp_v4_c3_detcov import fit as gp_fit, SIDX
from exp_dsx_lib import point_stats_robust, point_stats_parallel

DS = os.environ.get('DS', 'ds01')
assert DS in ('ds01', 'ds02', 'ds07', 'ds08a', 'ds04', 'ds05', 'ds06', 'ds08c')
SEEDS = [int(x) for x in os.environ.get('SEEDS_ONLY', '0,1,2').split(',')]
TRAIN_CYC_MAX, NPER_TRAIN, NPER = 3, 1000, 200
RES = os.path.join(HERE, f'{DS}v6_c3_results' + (f'_s{os.environ["SEEDS_ONLY"]}' if 'SEEDS_ONLY' in os.environ else '') + '.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'{DS.upper()} V6 C3 — stage-1 sparse-window MOGP, frozen internals '
        f'(rbf rank1 m=18, 5 sensors), dev only')
    cache = np.load(os.path.join(HERE, f'cache_{DS}.npz'))
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int); hs = A[:, 3]

    # per-unit data properties: hours, onset (mean-hs rule)
    props = {}
    for u in np.unique(unit):
        m = unit == u
        cyc_u = cyc[m]; hs_u = hs[m]
        ucyc = np.unique(cyc_u)
        hours = np.array([(cyc_u == c).sum() / 3600.0 for c in ucyc])
        hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
        below = np.where(hs_by < 0.5)[0]
        onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
        props[int(u)] = (ucyc, hours, onset)
        log(f'  u{u}: {len(ucyc)} cycles, mean {hours.mean():.2f} h/cycle, '
            f'hs-onset {onset}')

    for sd in SEEDS:
        out_path = os.path.join(HERE, f'{DS}v6_c3_stats_s{sd}.npz')
        if os.path.exists(out_path):
            log(f'seed{sd}: stats cached')
            continue
        rng = np.random.default_rng(sd)
        tr = []
        for u in np.unique(unit):
            for c in range(1, TRAIN_CYC_MAX + 1):
                rows = np.where((unit == u) & (cyc == c))[0]
                tr.append(rng.choice(rows, min(NPER_TRAIN, len(rows)),
                                     replace=False))
        tr = np.concatenate(tr)
        log(f'seed{sd}: train rows = {len(tr)} (cycle<={TRAIN_CYC_MAX}, '
            f'{NPER_TRAIN} per unit-cycle)')
        t0 = time.time()
        gp, nret = gp_fit(W[tr], X[tr])
        log(f'seed{sd}: fitted ({time.time()-t0:.0f}s, retries={nret})')
        gp_path = os.path.join(HERE, f'{DS}v6_gp_c3_s{sd}.pt')
        try:
            torch.save(gp, gp_path)
        except Exception as e:
            log(f'seed{sd}: model save FAILED ({e}) — continuing')
        out = {'train_idx': tr}
        for u in np.unique(unit):
            rows = np.where(unit == u)[0]
            cyc_u = cyc[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ucyc, hours, onset = props[int(u)]
            idx, cc = [], []
            for c in ucyc:
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                idx.append(r); cc.append(np.full(len(r), c))
            idx = np.concatenate(idx); cc = np.concatenate(cc)
            pred, dcv, ll = point_stats_parallel(gp, W[idx], X[idx], gp_path=gp_path, log=log)
            out[f'u{u}_cc'] = cc.astype(np.int32)
            out[f'u{u}_resid'] = (X[idx] - pred).astype(np.float32)
            out[f'u{u}_dc'] = dcv.astype(np.float32)
            out[f'u{u}_ll'] = ll.astype(np.float32)
            out[f'u{u}_ucyc'] = ucyc.astype(np.int32)
            out[f'u{u}_hours'] = hours.astype(np.float32)
            out[f'u{u}_onset'] = np.array([onset])
        np.savez(out_path, **out)
        log(f'seed{sd}: dev stats written ({time.time()-t0:.0f}s total)')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
