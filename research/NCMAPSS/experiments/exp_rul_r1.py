"""
exp_rul_r1.py — R1: build per-flight input tables for the degradation/RUL stage.

Per seed (0,1,2):
  1. rebuild the frozen 5-sensor final model (rbf r1, 5% detcov clean;
     same rng scheme as exp_sensor_study: 2000+sd / 3000+sd)
  2. gate threshold V* = 15th percentile of pooled DEV detcov (absolute value)
  3. for EVERY unit (dev 9 + test 6), every cycle, 200 pts (rng u*7+sd):
       per-point residual (5 sensors, raw units), detcov, fixed-LL
     -> per-flight rows:
       hours (cycle duration), cum_hours, kept count,
       gated mean / median / trim25 residual per sensor,
       q75 of gated LL (for detection in R5)
Saved to rul_input_s{sd}.npz:
  {split}_{u}_ucyc / _hours / _mean / _median / _trim / _llq75 / _kept / _onset
  plus Vstar
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from demo_cond2 import conditional_stats2
import gpytorch
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp

DT = torch.float32
SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
K = len(SENS)
SEEDS = [0, 1, 2]
NPTS, NCAND = 8192, 16384
M, STEPS, LR, MCOND, NPER, TRAIN_CYC, CLEAN, GATE_Q = 18, 120, 0.1, 15, 200, 5, 0.05, 15
RES = os.path.join(HERE, 'rul_r1_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def fit_gp(Wtr, Ytr):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT)
    tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=K, rank=1, kernel='rbf').to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=K).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=M, num_steps=STEPS, lr=LR,
                           group=True, verbose=False)
    model.eval(); lik.eval()
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure, xs=xs, ys=ys)


@torch.no_grad()
def point_stats(gp, Wq, Xq, chunk=4096):
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        p, dc, m2, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                               gp['tY'], gp['struct'], teX,
                                               m=MCOND, test_y=teY)
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


def trim25(v):
    s = np.sort(v, axis=0); n = len(v)
    return s[int(.25 * n):max(int(.25 * n) + 1, int(.75 * n))].mean(0)


def main():
    open(RES, 'w').close()
    log(f'R1: per-flight RUL input tables  sensors={SENS}  gate=V(q{GATE_Q}) abs  '
        f'summaries=[mean,median,trim25]  seeds={SEEDS}')
    cache = L.load_cache()
    data = {'dev': (cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']),
            'test': (cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test'])}
    W, X, A = data['dev']
    cyc_a = A[:, 1].astype(int)
    pool = np.where(cyc_a < TRAIN_CYC)[0]
    t00 = time.time()

    for sd in SEEDS:
        out_path = os.path.join(HERE, f'rul_input_s{sd}.npz')
        if os.path.exists(out_path):
            log(f'seed{sd}: exists, skip'); continue
        t0 = time.time()
        rng = np.random.default_rng(2000 + sd)
        cand = pool[rng.permutation(len(pool))[:NCAND]]
        tr1 = cand[rng.choice(NCAND, NPTS, replace=False)]
        gp1 = fit_gp(W[tr1], X[tr1])
        _, dc_c, _ = point_stats(gp1, W[cand], X[cand])
        keep = cand[dc_c >= np.percentile(dc_c, CLEAN * 100)]
        rng2 = np.random.default_rng(3000 + sd)
        tr = keep[rng2.choice(len(keep), NPTS, replace=False)]
        gp = fit_gp(W[tr], X[tr])
        log(f'seed{sd}: final model rebuilt ({time.time()-t0:.0f}s)')

        out = {}
        # V* from pooled dev detcov (computed on the fly, first pass over dev)
        dev_dc_pool = []
        unit_stats = {}
        for split in ['dev', 'test']:
            Wd, Xd, Ad = data[split]
            ua = Ad[:, 0].astype(int); ca = Ad[:, 1].astype(int); ha = Ad[:, 3]
            for u in np.unique(ua):
                rows = np.where(ua == u)[0]
                cyc_u = ca[rows]; hs_u = ha[rows]
                rng_u = np.random.default_rng(int(u) * 7 + sd)
                ucyc = np.unique(cyc_u)
                hours = np.array([(cyc_u == c).sum() / 3600.0 for c in ucyc])
                idx, cc = [], []
                for c in ucyc:
                    r = rows[cyc_u == c]
                    if len(r) > NPER:
                        r = rng_u.choice(r, NPER, replace=False)
                    idx.append(r); cc.append(np.full(len(r), c))
                idx = np.concatenate(idx); cc = np.concatenate(cc)
                pred, dcv, ll = point_stats(gp, Wd[idx], Xd[idx])
                resid = Xd[idx] - pred
                hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
                below = np.where(hs_by < 0.5)[0]
                onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
                unit_stats[(split, int(u))] = (ucyc, hours, cc, resid, dcv, ll, onset)
                if split == 'dev':
                    dev_dc_pool.append(dcv)
            log(f'seed{sd}: {split} point stats done ({time.time()-t0:.0f}s)')
        Vstar = float(np.percentile(np.concatenate(dev_dc_pool), GATE_Q))
        out['Vstar'] = np.array([Vstar])

        for (split, u), (ucyc, hours, cc, resid, dcv, ll, onset) in unit_stats.items():
            n = len(ucyc)
            mean_ = np.empty((n, K)); med_ = np.empty((n, K)); trm_ = np.empty((n, K))
            llq = np.empty(n); kept = np.empty(n, int)
            for i, c in enumerate(ucyc):
                mk = (cc == c) & (dcv >= Vstar)
                if not mk.any():
                    mk = cc == c
                r = resid[mk]
                mean_[i] = r.mean(0); med_[i] = np.median(r, 0); trm_[i] = trim25(r)
                llq[i] = np.percentile(ll[mk], 75); kept[i] = mk.sum()
            key = f'{split}_{u}'
            out[f'{key}_ucyc'] = ucyc.astype(np.int32)
            out[f'{key}_hours'] = hours.astype(np.float32)
            out[f'{key}_mean'] = mean_.astype(np.float32)
            out[f'{key}_median'] = med_.astype(np.float32)
            out[f'{key}_trim'] = trm_.astype(np.float32)
            out[f'{key}_llq75'] = llq.astype(np.float32)
            out[f'{key}_kept'] = kept.astype(np.int32)
            out[f'{key}_onset'] = np.array([onset])
        np.savez_compressed(out_path, **out)
        log(f'seed{sd}: saved {out_path}  V*={Vstar:.4f}  ({time.time()-t0:.0f}s total)')
    log(f'R1 DONE  wall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
