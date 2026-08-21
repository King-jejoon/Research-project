"""
exp_v4_bench_onset.py — benchmark, missing axis: ONSET DETECTION per normal model.

The RUL comparison showed the truncation metric cannot separate normal models.
Onset detection is the axis where the chain's advantage is supposed to live, so
it must be measured under the same component-swap rules.

Per model (llke / bspline / lr / cabn) and seed: fit on the seed's 27,000 training
rows, then compute the per-point corrected log-likelihood on the chain sampling
grid for all 9 dev units and all 6 test units (200 rows/cycle, rng u*7+sd).
Detector is byte-identical to the chain: per-cycle q75 of LL -> 30 flight-hour
baseline -> clip mu0 - 10 sigma0 -> full-range mean-drop.
Competitors run ungated (their leverage-based detcov is degenerate); mogp is
reported both gated at V*=20.2 (the chain) and ungated (fair comparison).
Score: onset RMSE = RMS(detected - true onset) over units x 3 seeds.

Protocol: uses the test statistics already opened for the benchmark batch; no
new opening.
Outputs: v4_bench_onset_results.txt, v4_bench_ll_s{sd}.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_bench2_lib as B2
from exp_v4_bench import train_idx_27k
from exp_v4_bench_rul import sample_grid, HP, MAKERS
from exp_v4_setcmp import detect

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
SEEDS = [0, 1, 2]
VSTAR = 20.2
RES = os.path.join(HERE, 'v4_bench_onset_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def build_ll(sd, cache):
    """per-point LL of every competitor on the dev and test sampling grids."""
    path = os.path.join(HERE, f'v4_bench_ll_s{sd}.npz')
    if os.path.exists(path):
        log(f'seed{sd}: LL cached')
        return
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    tr = train_idx_27k(sd, unit, cyc, np.where(cyc <= 3)[0])
    d_idx, d_cc, d_uu = sample_grid(A, sd)
    t_idx, t_cc, t_uu = sample_grid(At, sd)
    out = {'d_cc': d_cc, 'd_uu': d_uu, 't_cc': t_cc, 't_uu': t_uu}
    for name, maker in MAKERS.items():
        t0 = time.time()
        m = maker(**HP[name])
        np.random.seed(0)
        m.fit(W[tr], X[tr])
        _, _, ll_d = m.stats(W[d_idx], X[d_idx], chunk=4096)
        _, _, ll_t = m.stats(Wt[t_idx], Xt[t_idx], chunk=4096)
        out[f'{name}_ll_dev'] = ll_d.astype(np.float32)
        out[f'{name}_ll_test'] = ll_t.astype(np.float32)
        log(f'seed{sd} {name}: LL ready ({time.time()-t0:.0f}s)')
    np.savez(path, **out)


def onset_errors(name, gated=False):
    """detected - true onset over all units x seeds, for dev and test."""
    err = {'dev': [], 'test': []}
    for sd in SEEDS:
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
        Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
        if name != 'mogp':
            Zl = np.load(os.path.join(HERE, f'v4_bench_ll_s{sd}.npz'))
        for split in ['dev', 'test']:
            if split == 'dev':
                units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                                if k.endswith('_cc')})
            else:
                units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                                if k.endswith('_cc')})
            for u in units:
                if name == 'mogp':
                    Z = Zd if split == 'dev' else Zt
                    cc = Z[f'u{u}_cc']; ll = Z[f'u{u}_ll']; dc = Z[f'u{u}_dc']
                    true_on = int(Z[f'u{u}_onset'][0])
                else:
                    cc_all = Zl[f'{split[0] if split=="dev" else "t"}_cc'] \
                        if False else Zl['d_cc' if split == 'dev' else 't_cc']
                    uu_all = Zl['d_uu' if split == 'dev' else 't_uu']
                    ll_all = Zl[f'{name}_ll_{split}']
                    m = uu_all == u
                    cc = cc_all[m]; ll = ll_all[m]; dc = None
                    true_on = int((Zd if split == 'dev' else Zt)
                                  [f'u{u}_onset'][0])
                if split == 'dev':
                    ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
                    pos = {int(c): i for i, c in enumerate(ur)}
                    ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
                    dur = np.array([durs[pos[int(c)]] for c in ucyc], float)
                else:
                    ucyc = Zt[f'u{u}_ucyc']; dur = Zt[f'u{u}_hours'].astype(float)
                    keep = np.isin(ucyc, np.unique(cc))
                    ucyc, dur = ucyc[keep], dur[keep]
                cur = np.empty(len(ucyc))
                for i, c in enumerate(ucyc):
                    b = cc == c
                    mm = (b & (dc >= VSTAR)) if (gated and dc is not None) else b
                    if not mm.any():
                        mm = b
                    cur[i] = np.percentile(ll[mm], 75)
                err[split].append(detect(cur, dur, ucyc) - true_on)
    return {k: np.array(v, float) for k, v in err.items()}


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('BENCHMARK — onset detection per normal model (detector frozen: q75 LL, '
        '30 flight-h baseline, clip 10 sigma, mean-drop)')
    cache = L.load_cache()
    for sd in SEEDS:
        build_ll(sd, cache)

    rows = [('mogp gated V*=20.2', 'mogp', True),
            ('mogp ungated', 'mogp', False),
            ('llke', 'llke', False), ('bspline', 'bspline', False),
            ('lr', 'lr', False), ('cabn', 'cabn', False)]
    log('')
    log(f'  {"model":>20} | {"dev onset RMSE":>15} | {"dev delay":>10} | '
        f'{"test onset RMSE":>16} | {"test delay":>11} | test |d|<=3')
    store = {}
    for label, name, gated in rows:
        e = onset_errors(name, gated)
        store[label] = e
        d, t = e['dev'], e['test']
        log(f'  {label:>20} | {np.sqrt((d**2).mean()):15.2f} | '
            f'{d.mean():+10.2f} | {np.sqrt((t**2).mean()):16.2f} | '
            f'{t.mean():+11.2f} | {100*(np.abs(t)<=3).mean():.0f}%')
    np.savez(os.path.join(HERE, 'v4_bench_onset.npz'),
             **{f'{k}_{s}': v[s] for k, v in store.items()
                for s in ['dev', 'test']})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
