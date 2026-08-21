"""
exp_newclean_rul.py — PHASE 1d: per-flight RUL inputs under the raw-data
cleaning, so the HI -> RUL end of the chain can be checked too.

Same structure as exp_rul_r1.py (per-point residual / detcov / corrected LL,
gate at the absolute V* = 15th percentile of the pooled dev detcov, per-flight
mean / median / trim25 residual summaries and q75 of the gated LL), with one
difference: the final GP is trained on the pool cleaned by the model-free
robust neighbour score (score > 5 removed) instead of the stage-1 detcov 5 %
cleaning.  DEV UNITS ONLY — the DS03 test split stays sealed.

Output newclean_rul_s{seed}.npz mirrors the dev half of rul_input_s{seed}.npz
so the two can be fed to the identical HI/RUL evaluation.
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
from exp_traindata_detcov import fit_gp, SIDX, NPTS, NCAND, TRAIN_CYC, MCOND, DT
from demo_cond2 import conditional_stats2

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
K = len(SENS)
SEEDS = [0, 1, 2]
NPER, GATE_Q, CUT = 200, 15, 5.0
RES = os.path.join(HERE, 'newclean_rul_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def trim25(v):
    s = np.sort(v, axis=0); n = len(v)
    return s[int(.25 * n):max(int(.25 * n) + 1, int(.75 * n))].mean(0)


@torch.no_grad()
def point_stats(gp, Wq, Xq, chunk=4096):
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        p, dc, _, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                              gp['tY'], gp['struct'], teX,
                                              m=MCOND, test_y=teY)
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'PHASE 1d — per-flight RUL inputs with raw-data cleaning (score>{CUT})')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit_a = A[:, 0].astype(int); cyc_a = A[:, 1].astype(int); hs_a = A[:, 3]
    pool = np.where(cyc_a < TRAIN_CYC)[0]
    z = np.load(os.path.join(HERE, 'raw_clean.npz'))
    assert np.array_equal(z['pool'], pool)
    score = z['score']

    for sd in SEEDS:
        out_path = os.path.join(HERE, f'newclean_rul_s{sd}.npz')
        if os.path.exists(out_path):
            log(f'seed{sd}: cached'); continue
        t0 = time.time()
        rng = np.random.default_rng(2000 + sd)
        cand = pool[rng.permutation(len(pool))[:NCAND]]
        _ = cand[rng.choice(NCAND, NPTS, replace=False)]
        keep = cand[score[np.searchsorted(pool, cand)] <= CUT]
        rng2 = np.random.default_rng(3000 + sd)
        tr = keep[rng2.choice(len(keep), NPTS, replace=False)]
        gp = fit_gp(W[tr], X[tr])
        log(f'seed{sd}: model built ({time.time()-t0:.0f}s)')

        unit_stats = {}
        dc_pool = []
        for u in np.unique(unit_a):
            rows = np.where(unit_a == u)[0]
            cyc_u = cyc_a[rows]; hs_u = hs_a[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ucyc = np.unique(cyc_u)
            idx, cc, hours = [], [], []
            for c in ucyc:
                r = rows[cyc_u == c]
                hours.append(len(r) / 3600.0)          # 1 Hz rows -> flight hours
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                idx.append(r); cc.append(np.full(len(r), c))
            idx = np.concatenate(idx); cc = np.concatenate(cc)
            pred, dcv, ll = point_stats(gp, W[idx], X[idx])
            resid = X[idx] - pred
            hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
            below = np.where(hs_by < 0.5)[0]
            onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
            unit_stats[u] = (ucyc, np.array(hours), cc, resid, dcv, ll, onset)
            dc_pool.append(dcv)
        Vstar = float(np.percentile(np.concatenate(dc_pool), GATE_Q))
        log(f'seed{sd}: point stats done, V*={Vstar:.4f} ({time.time()-t0:.0f}s)')

        out = {'Vstar': np.array([Vstar])}
        for u, (ucyc, hours, cc, resid, dcv, ll, onset) in unit_stats.items():
            n = len(ucyc)
            mean_ = np.empty((n, K)); med_ = np.empty((n, K)); trm_ = np.empty((n, K))
            llq = np.empty(n); kept = np.empty(n, int)
            mean_ng = np.empty((n, K)); med_ng = np.empty((n, K))
            trm_ng = np.empty((n, K)); llq_ng = np.empty(n)
            for i, c in enumerate(ucyc):
                base = cc == c
                mk = base & (dcv >= Vstar)
                if not mk.any():
                    mk = base
                r = resid[mk]
                mean_[i] = r.mean(0); med_[i] = np.median(r, 0); trm_[i] = trim25(r)
                llq[i] = np.percentile(ll[mk], 75); kept[i] = mk.sum()
                # ungated twins, for the gate on/off ablation downstream
                rn = resid[base]
                mean_ng[i] = rn.mean(0); med_ng[i] = np.median(rn, 0)
                trm_ng[i] = trim25(rn)
                llq_ng[i] = np.percentile(ll[base], 75)
            key = f'dev_{u}'
            out[f'{key}_ucyc'] = ucyc.astype(np.int32)
            out[f'{key}_hours'] = hours.astype(np.float32)
            out[f'{key}_mean'] = mean_.astype(np.float32)
            out[f'{key}_median'] = med_.astype(np.float32)
            out[f'{key}_trim'] = trm_.astype(np.float32)
            out[f'{key}_llq75'] = llq.astype(np.float32)
            out[f'{key}_mean_ng'] = mean_ng.astype(np.float32)
            out[f'{key}_median_ng'] = med_ng.astype(np.float32)
            out[f'{key}_trim_ng'] = trm_ng.astype(np.float32)
            out[f'{key}_llq75_ng'] = llq_ng.astype(np.float32)
            out[f'{key}_kept'] = kept.astype(np.int32)
            out[f'{key}_onset'] = np.array([onset])
        np.savez(out_path, **out)
        log(f'seed{sd}: saved ({time.time()-t0:.0f}s total)')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
