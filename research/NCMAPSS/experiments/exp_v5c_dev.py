"""
exp_v5c_dev.py — PROPOSED healthy-range chain, dev design (no test scoring).

User decisions (2026-08-14): training window = all cycles strictly before
the CURRENT conditional-gate onset (the official frozen chain's own
detection: V = 19.0 if nbase < 10 else 20.2, q75 corrected-LL curve,
30 flight-h baseline, clip mu0 - 10 sigma0, mean-drop), sample size
matched to the official chain (3,000 rows per unit stratified equally
over healthy cycles, 27k total, 3 seeds), and the downstream runs
UNGATED — no detcov gate anywhere after the residuals.

Phase 0 guards (cached official stats only, before any GP fit):
  (g1) conditional onsets recomputed from v4_c3_stats must reproduce the
       official dev onset RMSE 5.21 (ungated reference 6.78, logged);
  (g2) if the conditional onsets equal the old fixed-gate onsets
       (exp_v4_exp2, V*=20.2) for every unit and seed, ABORT — the
       window would duplicate the recorded exp2 ablation.

Chain per seed: sample window rows (rng 3000+sd) -> refit MOGP (frozen
internals: rbf rank1, 5 sensors) -> SAVE the model -> dev point stats
(NPER=200, rng u*7+sd, chain convention; ll saved as well) -> UNGATED
tables -> trim25 -> pooled z -> HI l1=12 l2=0.25 + med3 -> beta by the
frozen dev-LOO NASA rule -> dev LOO truncation RUL.

References for judgement (recorded, not re-run): official re-frozen
chain dev 7.22 +/- 0.08 (v5b_dev.npz, paired on the 108 predictions);
old healthy-range ablation dev 7.61 (mean only, preds not recorded).
Outputs: v5c_dev_results.txt, v5c_stats_s{sd}.npz, v5c_gp_s{sd}.pt,
         v5c_dev.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
import exp_v4_exp2 as E2                     # old fixed-gate onsets (g2)
from exp_v4_c3_detcov import fit as gp_fit, point_stats, SIDX
from exp_v4_setcmp import load as load_stats, detect
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, train_model_tail, \
    shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3

SEEDS = [0, 1, 2]
VD, VS, NB = 20.2, 19.0, 10
NPER, NTOT_PER_UNIT = 200, 3000
L1, L2 = 12.0, 0.25
RES = os.path.join(HERE, 'v5c_dev_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def cond_onsets(sd):
    """official conditional-gate onsets from cached stats + onset errors."""
    D = load_stats(5, sd)
    out, errs = {}, []
    for u, d in D.items():
        nb = max(5, int(np.searchsorted(np.cumsum(d['dur']), 30.0) + 1))
        V = VS if nb < NB else VD
        cur = np.empty(len(d['ucyc']))
        for i, c in enumerate(d['ucyc']):
            b = d['cc'] == c
            m = b & (d['dc'] >= V)
            if not m.any():
                m = b
            cur[i] = np.percentile(d['ll'][m], 75)
        out[u] = detect(cur, d['dur'], d['ucyc'])
        errs.append(out[u] - d['onset'])
    return out, np.array(errs, float)


def ungated_errs(sd):
    D = load_stats(5, sd)
    e = []
    for u, d in D.items():
        cur = np.array([np.percentile(d['ll'][d['cc'] == c], 75)
                        for c in d['ucyc']])
        e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
    return np.array(e, float)


def phase0():
    ons, ndiff = {}, 0
    ce, ue = [], []
    for sd in SEEDS:
        o, e = cond_onsets(sd)
        ons[sd] = o
        ce.append(e)
        ue.append(ungated_errs(sd))
        old = E2.detected_onsets(sd)
        d = {u: (old[u], o[u]) for u in o if old[u] != o[u]}
        ndiff += len(d)
        log(f'seed{sd}: conditional onsets {o}')
        log(f'seed{sd}: differs from old fixed-gate at {len(d)}/{len(o)} '
            'units: ' + (', '.join(f'u{u} {a}->{b}' for u, (a, b)
                                   in sorted(d.items())) or 'none'))
    rc = float(np.sqrt((np.concatenate(ce) ** 2).mean()))
    ru = float(np.sqrt((np.concatenate(ue) ** 2).mean()))
    log(f'g1 sanity: conditional onset RMSE = {rc:.2f} (expected 5.21), '
        f'ungated = {ru:.2f} (reference 6.78)')
    assert abs(rc - 5.21) <= 0.02, 'g1 FAILED — conditional onset RMSE off'
    assert ndiff > 0, 'g2 ABORT — windows identical to exp2 everywhere'
    return ons


def build_stats(sd, cache, ons):
    out_path = os.path.join(HERE, f'v5c_stats_s{sd}.npz')
    want = np.array([[u, ons[u]] for u in sorted(ons)], int)
    if os.path.exists(out_path):
        got = np.load(out_path)['onset_used']
        assert np.array_equal(got, want), \
            f'seed{sd}: cached v5c_stats built from different onsets — delete'
        log(f'seed{sd}: stats cached (onsets match)')
        return
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    rng = np.random.default_rng(3000 + sd)
    tr = []
    for u in sorted(ons):
        o = ons[u]
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
        log(f'seed{sd}: u{u} window cyc<{o} ({len(ucyc)} cycles, pool '
            f'{len(rows_u)} rows) -> {len(got)} sampled')
        tr.append(got)
    tr = np.concatenate(tr)
    log(f'seed{sd}: train total {len(tr)} rows')
    t0 = time.time()
    gp, nret = gp_fit(W[tr], X[tr])
    log(f'seed{sd}: fitted ({time.time()-t0:.0f}s, retries={nret})')
    try:
        torch.save(gp, os.path.join(HERE, f'v5c_gp_s{sd}.pt'))
        log(f'seed{sd}: model saved')
    except Exception as e:                      # save is best-effort
        log(f'seed{sd}: model save FAILED ({e}) — continuing')
    out = {'train_idx': tr, 'onset_used': want}
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
        out[f'u{u}_ll'] = ll.astype(np.float32)
    np.savez(out_path, **out)
    log(f'seed{sd}: dev stats written ({time.time()-t0:.0f}s total)')


def tables(sd):
    """UNGATED per-cycle tables + pooled z-norm (proposed downstream)."""
    Z = np.load(os.path.join(HERE, f'v5c_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Z[f'u{u}_cc']; rs = Z[f'u{u}_resid']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[cc == c])
        raw[u] = T
        hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, raw, {u: (raw[u] - mu) / sg for u in units}, hrs, mu, sg


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5C DEV — proposed healthy-range chain (conditional-gate onsets '
        'define the window; downstream UNGATED; user decisions 2026-08-14)')
    ons = phase0()
    cache = L.load_cache()
    for sd in SEEDS:
        build_stats(sd, cache, ons[sd])

    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    rms, nss, oks, bs, Ps, Ts, Fs = [], [], [], [], [], [], []
    for sd in SEEDS:
        units, raw, Zn, hrs, mu, sg = tables(sd)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok)); bs.append(b)
        Ps.append(P); Ts.append(T); Fs.append(F)
        log(f'seed{sd}: dev LOO={rms[-1]:.2f} '
            f'shape={"OK" if ok else "FAIL"} beta={b}')
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log('')
    log(f'V5C dev LOO = {np.mean(rms):.2f} ± {np.std(rms):.2f}  '
        f'NASA {np.mean(nss):.1f} ± {np.std(nss):.1f}  '
        f'shape {int(np.sum(oks))}/3  beta={bs}')
    log('dev per-trunc = ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    Z = np.load(os.path.join(HERE, 'v5b_dev.npz'))
    assert np.allclose(T, Z['l12_T']), 'pairing misaligned vs v5b_dev'
    pw = st.wilcoxon(np.abs(P - T), np.abs(Z['l12_P'] - Z['l12_T']),
                     zero_method='zsplit').pvalue
    log(f'paired |err| vs official chain 7.22 (108 preds): p={pw:.3f}')
    log('reference: old healthy-range ablation dev 7.61 (mean only)')
    np.savez(os.path.join(HERE, 'v5c_dev.npz'), P=P, T=T, F=F,
             rms=np.array(rms), nss=np.array(nss), bs=np.array(bs),
             onsets=np.array([[sd, u, ons[sd][u]] for sd in SEEDS
                              for u in sorted(ons[sd])], int))
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
