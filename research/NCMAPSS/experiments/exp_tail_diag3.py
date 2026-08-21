"""
exp_tail_diag3.py — PHASE 0c: WHY do units 6/7/8 dominate the healthy error?

Phase 0b: unit 7 alone carries 73.7% of the total squared healthy error
(mean |z| 0.72 vs 0.26-0.29 for units 1-5), and the cross-unit neighbour
correlation stays at 0.82.  Three competing explanations, each with a
different consequence for data cleaning:

  (A) per-unit baseline offset — every engine has its own healthy signature and
      the POOLED normal model cannot represent it.  Signature: the signed
      residual has a large, roughly constant mean per unit/sensor.
      => not removable dirt; a modelling issue (per-unit centering).
  (B) early degradation before the hs label flips — the "healthy" evaluation
      points of those units are already drifting.  Signature: |z| grows with
      cycle within the unit, pre-onset.
      => not dirt either; it is genuine signal (and would HELP detection).
  (C) operating-envelope mismatch — those units fly a different flight class,
      so their W lies outside the training coverage.  Signature: elevated error
      concentrated in a flight class under-represented in cycle<5.

Uses the residuals cached by exp_tail_diag.py; no GP fit.
"""
import os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
from exp_traindata_detcov import true_onset_mask, NPTS, NCAND, NEVAL, TRAIN_CYC

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
RES = os.path.join(HERE, 'tail_diag3_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    d = np.load(os.path.join(HERE, 'tail_diag.npz'))
    zn, z = d['zn'], d['z']          # |z| per point, signed z per sensor
    cache = L.load_cache()
    W, A = cache['W_dev'], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy_gen = np.where(true_onset_mask(A))[0]

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pool))
    cand = pool[perm[:NCAND]]
    _evin = pool[perm[NCAND:NCAND + NEVAL]]
    evgn = rng.choice(healthy_gen, NEVAL, replace=False)

    unit = A[evgn, 0].astype(int)
    cycle = A[evgn, 1].astype(int)
    fc = A[evgn, 2].astype(int)

    log('PHASE 0c — why do units 6/7/8 dominate the healthy error?')
    log('')
    log('(A) per-unit SIGNED residual mean  (bias, in sigma units)')
    log(f'   {"unit":>4} | ' + ' | '.join(f'{s:>7}' for s in SENS) +
        ' | mean|bias| | RMS around bias')
    for u in np.unique(unit):
        m = unit == u
        b = z[m].mean(0)
        resid_dm = z[m] - b
        log(f'   {u:>4} | ' + ' | '.join(f'{v:>7.3f}' for v in b) +
            f' | {np.abs(b).mean():>10.3f} | {np.sqrt((resid_dm**2).mean()):>15.3f}')
    tot = (z ** 2).sum()
    bias_part = sum(((z[unit == u].mean(0)) ** 2 * (unit == u).sum()).sum()
                    for u in np.unique(unit))
    log(f'   -> share of total z^2 explained by a per-unit constant offset: '
        f'{100*bias_part/tot:.1f}%')

    log('')
    log('(B) trend of |z| with cycle, pre-onset (early degradation?)')
    log(f'   {"unit":>4} | {"n":>5} | {"cyc range":>10} | {"Spearman":>9} | '
        f'{"p":>9} | {"|z| first-third":>15} | {"last-third":>10}')
    for u in np.unique(unit):
        m = unit == u
        c, v = cycle[m], zn[m]
        rho, p = stats.spearmanr(c, v)
        lo, hi = np.percentile(c, [33, 67])
        log(f'   {u:>4} | {m.sum():>5} | {c.min():>4}-{c.max():>4} | '
            f'{rho:>9.3f} | {p:>9.2e} | {v[c <= lo].mean():>15.3f} | '
            f'{v[c >= hi].mean():>10.3f}')

    log('')
    log('(C) flight class composition and error')
    log(f'   {"unit":>4} | Fc counts | mean|z| by Fc')
    for u in np.unique(unit):
        m = unit == u
        cnt = {int(f): int((fc[m] == f).sum()) for f in np.unique(fc[m])}
        byf = {int(f): round(float(zn[m & (fc == f)].mean()), 3)
               for f in np.unique(fc[m])}
        log(f'   {u:>4} | {cnt} | {byf}')
    log('')
    log('   training pool (cycle<5) flight-class composition:')
    fcp = A[pool, 2].astype(int)
    log(f'     {dict(zip(*[x.tolist() for x in np.unique(fcp, return_counts=True)]))}')
    log('   healthy-gen eval flight-class composition:')
    log(f'     {dict(zip(*[x.tolist() for x in np.unique(fc, return_counts=True)]))}')


if __name__ == '__main__':
    main()
