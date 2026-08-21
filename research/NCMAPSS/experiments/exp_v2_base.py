"""
exp_v2_base.py — FRESH START (v2 chain), step 1: baseline normal model.

Configuration set by the user:
  * sensors  : T30, T48, T50, Nc, Wf  (5)
  * kernel   : RBF, coregionalization rank 1
  * training rows : cycle < 3 of every dev unit
  * sampling : 1000 rows per (unit, cycle)  -> 9 units x 2 cycles x 1000 = 18,000
GP internals unchanged from the frozen family: Vecchia m=18, Adam lr 0.1 x 120,
StandardScaler on inputs/outputs, inputs W = [alt, Mach, TRA, T2].

This script only trains and characterises the model (3 seeds):
  eval_in    8192 held-out rows from cycle<3 (excluded from training)
  eval_early 8192 healthy rows from cycles 3..8 (pre-onset)
Reported per set: zRMSE (sigma = this model's residual s.d. on that set is NOT
used; sigma is fixed from seed-0's model so seeds are comparable), trimmed
zRMSE (central 90%), median |z|, and per-sensor raw RMSE.
Training indices per seed are stored in v2_base_train_s{seed}.npz so every
later step can rebuild the identical model.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import fit_gp, predict, true_onset_mask, SIDX

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
TRAIN_CYC = 3                 # cycle < 3
NPER_TRAIN = 1000             # rows per (unit, cycle)
NEVAL = 8192
EARLY_HI = 8
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'v2_base_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def stats(r, sig):
    z = r / sig
    zn = np.sqrt((z ** 2).mean(1))
    keep = zn <= np.percentile(zn, 90)
    return (float(np.sqrt((z ** 2).mean())),
            float(np.sqrt((z[keep] ** 2).mean())),
            float(np.median(zn)))


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('V2 BASELINE MODEL — 5 sensors, RBF rank1, cycle<3, 1000 rows/(unit,cycle)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy = true_onset_mask(A)          # healthy uses its own TRAIN_CYC=5 floor
    early = np.where(healthy & (cyc >= TRAIN_CYC) & (cyc <= EARLY_HI))[0]
    log(f'  pool cycle<{TRAIN_CYC}: {len(pool)} rows | '
        f'early (cyc {TRAIN_CYC}-{EARLY_HI}, pre-onset): {len(early)} rows')

    sig_ref = {}
    for sd in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(sd)
        # ---- per-(unit,cycle) stratified draw: 1000 rows each ----
        tr = []
        for u in np.unique(unit[pool]):
            for c in range(1, TRAIN_CYC):
                rows = pool[(unit[pool] == u) & (cyc[pool] == c)]
                take = min(NPER_TRAIN, len(rows))
                tr.append(rng.choice(rows, take, replace=False))
        tr = np.concatenate(tr)
        rest = np.setdiff1d(pool, tr, assume_unique=False)
        evin = rng.choice(rest, min(NEVAL, len(rest)), replace=False)
        evearly = rng.choice(early, NEVAL, replace=False)
        log(f'seed{sd}: train={len(tr)} rows '
            f'({len(np.unique(unit[tr]))} units x {TRAIN_CYC-1} cycles x {NPER_TRAIN})')

        gp = fit_gp(W[tr], X[tr])
        tfit = time.time() - t0
        log(f'seed{sd}: fitted ({tfit:.0f}s)  noise={gp["noise"]:.5f}')

        for snm, ev in [('in', evin), ('early', evearly)]:
            r = X[ev] - predict(gp, W[ev])
            if sd == 0:
                sig_ref[snm] = r.std(0)
            zr, zt, zm = stats(r, sig_ref[snm])
            per = np.sqrt((r ** 2).mean(0))
            log(f'seed{sd} eval_{snm:>5}: zRMSE={zr:.4f} trim={zt:.4f} med={zm:.4f} | '
                + ' '.join(f'{s}={v:.3f}' for s, v in zip(SENS, per)))
        np.savez(os.path.join(HERE, f'v2_base_train_s{sd}.npz'),
                 train_idx=tr, evin=evin, evearly=evearly,
                 sig_in=sig_ref['in'], sig_early=sig_ref['early'])
        log(f'seed{sd}: done ({time.time()-t0:.0f}s total)')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
