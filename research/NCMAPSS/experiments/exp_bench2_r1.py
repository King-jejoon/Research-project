"""exp_bench2_r1.py — residual tables for the NORMAL-MODEL benchmark.

For every competing normal model (lr / bspline / llke / cabn) this reproduces
exactly what exp_rul_r1.py does for the Vecchia MOGP:

  1. hyper-parameters chosen on DEV held-out healthy zRMSE (stage-appropriate
     metric: this block is judged as a predictor of healthy sensor values)
  2. per seed: 16384 candidates from dev cycle<5 -> fit -> drop the 5 % lowest
     detcov -> refit on 8192 points          (identical cleaning rule)
  3. point statistics for every unit, 200 points per cycle, rng(u*7+sd)
  4. V* = 15th percentile of the pooled dev detcov  (own value, absolute)
  5. per-flight gated mean / median / trim25 residual + q75 of gated LL

Saved to bench2_{method}_s{sd}.npz with the same keys as rul_input_s{sd}.npz,
so the frozen downstream (HI network, first-passage RUL, onset detector) reads
them without modification.  Timings are recorded in bench2_times.json.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
import exp_bench2_lib as B2

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
K = len(SENS)
SEEDS = [0, 1, 2]
NPTS, NCAND = 8192, 16384
NPER, TRAIN_CYC, CLEAN, GATE_Q = 200, 5, 0.05, 15
RES = os.path.join(HERE, 'bench2_r1_results.txt')

GRIDS = {
    'lr':      [dict()],
    'bspline': [dict(n_knots=k) for k in (3, 5, 8, 12, 20, 30)],
    'llke':    [dict(h=h) for h in (0.05, 0.1, 0.15, 0.2, 0.35, 0.5)],
    'cabn':    [dict(lam1=l) for l in (0.0, 0.001, 0.01, 0.05)],
}


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def trim25(v):
    s = np.sort(v, axis=0); n = len(v)
    return s[int(.25 * n):max(int(.25 * n) + 1, int(.75 * n))].mean(0)


def select_hp(name, W, X, pool):
    """dev-only selection on held-out HEALTHY MEAN LOG-LIKELIHOOD.

    zRMSE only sees the conditional mean, so it cannot separate models that
    differ in their covariance -- CaBN's sparse DAG changes Phi but not the
    mean, and every lam1 would tie.  The log-likelihood is the proper scoring
    rule for a conditional density and is what the downstream gate (detcov) and
    onset detector (LL) actually consume.  zRMSE is reported alongside."""
    best = None
    for hp in GRIDS[name]:
        zs, ls = [], []
        for sd in SEEDS:
            rng = np.random.default_rng(2000 + sd)
            cand = pool[rng.permutation(len(pool))[:NCAND]]
            tr = cand[rng.choice(NCAND, NPTS, replace=False)]
            ho = cand[~np.isin(cand, tr)]
            m = B2.build(name, **hp)
            m.fit(W[tr], X[tr])
            pred, _, ll = m.stats(W[ho], X[ho])
            zs.append(float(np.sqrt(((m.ys.transform(X[ho]) -
                                      m.ys.transform(pred)) ** 2).mean())))
            ls.append(float(np.mean(ll)))
        z, l = float(np.mean(zs)), float(np.mean(ls))
        log(f'  {name:>8} {str(hp):<20} held-out LL={l:9.3f}  zRMSE={z:.4f}')
        if best is None or l > best[0]:
            best = (l, hp, z)
    log(f'  -> {name} selected {best[1]}  (LL {best[0]:.3f}, zRMSE {best[2]:.4f})')
    return best[1], dict(ll=best[0], zrmse=best[2])


def main():
    open(RES, 'w').close()
    log('BENCH2 R1: residual tables for the normal-model benchmark')
    log(f'  sensors={SENS}  inputs=[alt,Mach,TRA,T2]  seeds={SEEDS}  '
        f'clean={CLEAN:.0%}  gate=q{GATE_Q}  {NPER} pts/cycle')
    cache = L.load_cache()
    data = {'dev': (cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']),
            'test': (cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test'])}
    W, X, A = data['dev']
    pool = np.where(A[:, 1].astype(int) < TRAIN_CYC)[0]
    times = {}
    t00 = time.time()

    for name in GRIDS:
        log(f'\n=== {name} ===')
        hp, qual = select_hp(name, W, X, pool)
        times[name] = dict(hp=hp, **qual, fit=[], infer_dev=[], infer_test=[])

        for sd in SEEDS:
            out_path = os.path.join(HERE, f'bench2_{name}_s{sd}.npz')
            t0 = time.time()
            rng = np.random.default_rng(2000 + sd)
            cand = pool[rng.permutation(len(pool))[:NCAND]]
            tr1 = cand[rng.choice(NCAND, NPTS, replace=False)]
            m1 = B2.build(name, **hp)
            m1.fit(W[tr1], X[tr1])
            _, dc_c, _ = m1.stats(W[cand], X[cand])
            keep = cand[dc_c >= np.percentile(dc_c, CLEAN * 100)]
            rng2 = np.random.default_rng(3000 + sd)
            tr = keep[rng2.choice(len(keep), NPTS, replace=False)]
            model = B2.build(name, **hp)
            model.fit(W[tr], X[tr])
            t_fit = time.time() - t0
            log(f'  seed{sd}: fitted ({t_fit:.1f}s)')

            out, unit_stats, dev_dc = {}, {}, []
            t_inf = {}
            for split in ['dev', 'test']:
                Wd, Xd, Ad = data[split]
                ua = Ad[:, 0].astype(int); ca = Ad[:, 1].astype(int); ha = Ad[:, 3]
                t1 = time.time()
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
                    pred, dcv, ll = model.stats(Wd[idx], Xd[idx])
                    resid = Xd[idx] - pred
                    hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
                    below = np.where(hs_by < 0.5)[0]
                    onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
                    unit_stats[(split, int(u))] = (ucyc, hours, cc, resid, dcv, ll, onset)
                    if split == 'dev':
                        dev_dc.append(dcv)
                t_inf[split] = time.time() - t1
                log(f'  seed{sd}: {split} point stats done ({t_inf[split]:.1f}s)')

            Vstar = float(np.percentile(np.concatenate(dev_dc), GATE_Q))
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
            times[name]['fit'].append(t_fit)
            times[name]['infer_dev'].append(t_inf['dev'])
            times[name]['infer_test'].append(t_inf['test'])
            log(f'  seed{sd}: saved {os.path.basename(out_path)}  V*={Vstar:.3f}  '
                f'kept={np.mean([out[k].mean() for k in out if k.endswith("_kept")]):.0f}/200 '
                f'({time.time()-t00:.0f}s elapsed)')

    with open(os.path.join(HERE, 'bench2_times.json'), 'w') as f:
        json.dump(times, f, indent=1)
    log(f'\nBENCH2 R1 DONE  wall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
