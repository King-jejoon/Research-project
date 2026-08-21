"""
exp_v4_exp2.py — sensitivity experiment 2: healthy-range retraining chain.

The chain's own flight-hour detector (V*=20.2, q75, 30 flight-h baseline,
clip 10 sigma, mean-drop) already produced per-unit onsets on dev.  This
experiment retrains the SAME model (rbf rank1, 5 sensors, frozen internals)
on the DETECTED healthy range instead of the fixed cycle<=3 window:

  training pool (per seed) = for every dev unit, all cycles BEFORE its
  detected onset (self-supervised: detected, not hs-labelled);
  sample size controlled at 27,000 (3,000 per unit, stratified equally
  over its healthy cycles) so coverage is the only variable.

Everything downstream is identical (trim25 -> HI l1=8 l2=0.25 med3 ->
beta dev-LOO); onset scoring is SKIPPED (the range came from the detector),
the chain is scored by dev LOO truncation RUL only.  The gate is re-swept on
this model's own absolute detcov axis — coverage-dependence check: with full
healthy coverage the gate is expected to lose its value.
Outputs: v4_exp2_results.txt, v4_exp2_stats_s{sd}.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from exp_v4_c3_detcov import fit as gp_fit, point_stats, SIDX
from exp_v4_setcmp import load as load_stats, detect
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3

SEEDS = [0, 1, 2]
VSTAR_OLD = 20.2
NPER, NTOT_PER_UNIT = 200, 3000
L1, L2 = 8.0, 0.25
RES = os.path.join(HERE, 'v4_exp2_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def detected_onsets(sd):
    D = load_stats(5, sd)
    out = {}
    for u, d in D.items():
        cur = np.empty(len(d['ucyc']))
        for i, c in enumerate(d['ucyc']):
            b = d['cc'] == c
            m = b & (d['dc'] >= VSTAR_OLD)
            if not m.any():
                m = b
            cur[i] = np.percentile(d['ll'][m], 75)
        out[u] = detect(cur, d['dur'], d['ucyc'])
    return out


def build_stats(sd, cache):
    out_path = os.path.join(HERE, f'v4_exp2_stats_s{sd}.npz')
    if os.path.exists(out_path):
        log(f'seed{sd}: stats cached')
        return
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    ons = detected_onsets(sd)
    log(f'seed{sd}: detected onsets {ons}')
    rng = np.random.default_rng(1000 + sd)
    tr = []
    for u, o in ons.items():
        rows_u = np.where((unit == u) & (cyc < o))[0]
        cyc_u = cyc[rows_u]
        ucyc = np.unique(cyc_u)
        per = int(np.ceil(NTOT_PER_UNIT / len(ucyc)))
        got = []
        for c in ucyc:
            r = rows_u[cyc_u == c]
            got.append(rng.choice(r, min(per, len(r)), replace=False))
        got = np.concatenate(got)
        if len(got) > NTOT_PER_UNIT:
            got = rng.choice(got, NTOT_PER_UNIT, replace=False)
        tr.append(got)
    tr = np.concatenate(tr)
    log(f'seed{sd}: healthy-range train={len(tr)} rows '
        f'({len(ons)} units x ~{NTOT_PER_UNIT})')
    t0 = time.time()
    gp, nret = gp_fit(W[tr], X[tr])
    log(f'seed{sd}: fitted ({time.time()-t0:.0f}s, retries={nret})')
    out = {'train_idx': tr}
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]
        cyc_u = cyc[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        ucyc = np.unique(cyc_u)
        idx, cc = [], []
        for c in ucyc:
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx.append(r); cc.append(np.full(len(r), c))
        idx = np.concatenate(idx); cc = np.concatenate(cc)
        pred, dcv, ll = point_stats(gp, W[idx], X[idx])
        out[f'u{u}_cc'] = cc.astype(np.int32)
        out[f'u{u}_resid'] = (X[idx] - pred).astype(np.float32)
        out[f'u{u}_dc'] = dcv.astype(np.float32)
    np.savez(out_path, **out)
    log(f'seed{sd}: stats cached ({time.time()-t0:.0f}s total)')


def tables(sd, V):
    Z = np.load(os.path.join(HERE, f'v4_exp2_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Z[f'u{u}_cc']; rs = Z[f'u{u}_resid']; dc = Z[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= V)
            T[i] = trim25(rs[m if m.any() else b])
        raw[u] = T
        hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, {u: (raw[u] - mu) / sg for u in units}, hrs


def rul_at(V):
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    rms, nss, errs, oks = [], [], [], []
    for sd in SEEDS:
        units, Zn, hrs = tables(sd, V)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); errs.append(P - T); oks.append(int(ok))
    return (float(np.mean(rms)), float(np.std(rms)), float(np.mean(nss)),
            int(np.sum(oks)), np.concatenate(errs))


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('SENSITIVITY 2 — healthy-range retraining chain (detected onsets '
        'define the window; onset scoring skipped; dev LOO RUL only)')
    cache = L.load_cache()
    for sd in SEEDS:
        build_stats(sd, cache)

    # detcov scale of the new model
    pooled = []
    for sd in SEEDS:
        Z = np.load(os.path.join(HERE, f'v4_exp2_stats_s{sd}.npz'))
        pooled.append(np.concatenate([Z[k] for k in Z.files
                                      if k.endswith('_dc')]))
    pooled = np.concatenate(pooled)
    qs = [1, 2, 5, 10, 25, 50, 75, 90, 99]
    log('pooled detcov (descriptive): '
        + '  '.join(f'p{q}={np.percentile(pooled, q):.2f}' for q in qs))

    r0 = rul_at(-1e9)
    log(f'gate off : RUL RMSE={r0[0]:.2f} ± {r0[1]:.2f}  NASA={r0[2]:.1f}  '
        f'shape {r0[3]}/3   [cycle<=3 chain reference: 7.30 ± 0.16]')
    lo = np.floor(np.percentile(pooled, 2) * 10) / 10
    hi = np.ceil(np.percentile(pooled, 98) * 10) / 10
    grid = np.round(np.arange(lo, hi + 1e-9, 0.1), 1)
    log(f'absolute V sweep {grid[0]}..{grid[-1]} step 0.1 ({len(grid)} values)')
    best = None
    for V in grid:
        r = rul_at(float(V))
        kept = 100.0 * (pooled >= V).mean()
        pw = st.wilcoxon(np.abs(r[4]), np.abs(r0[4]),
                         zero_method='zsplit').pvalue
        log(f'  V={V:5.1f} ({kept:5.1f}% kept): RUL RMSE={r[0]:.2f} ± '
            f'{r[1]:.2f}  NASA={r[2]:.1f}  shape {r[3]}/3  p={pw:.3f}')
        if best is None or r[0] < best[1]:
            best = (float(V), r[0], pw)
    log('')
    log(f'best V={best[0]} (RUL RMSE {best[1]:.2f}, p={best[2]:.3f}) vs '
        f'gate off {r0[0]:.2f} — coverage-dependence check')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
