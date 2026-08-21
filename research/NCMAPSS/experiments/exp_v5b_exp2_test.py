"""
exp_v5b_exp2_test.py — healthy-range-window ablation completed on test:
addendum to the ELEVENTH DS03 opening (pre-registered scope: finish the
Section-10 ablation tables on test; this variant was deferred because its
GP was never saved).

Per seed: refit the healthy-range GP exactly as exp_v4_exp2 (window =
detected onsets at the old fixed gate, 27k rows, same rng), SAVE the model
this time (torch.save), compute test residual/detcov stats (NPER=200,
rng u*7+sd — chain convention), then the frozen downstream: gate off,
HI l1=12 l2=0.25 + med3, beta dev-LOO NASA -> test truncation RUL.
Sanity: dev LOO must reproduce 7.61 +/- 0.02 (v5b_ablates section 4).
Outputs: v5b_exp2_test_results.txt, v4_exp2_test_stats_s{sd}.npz,
         v4_exp2_gp_s{sd}.pt, v5b_exp2_test.npz
"""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_v4_exp2 as E2
from exp_v4_c3_detcov import fit as gp_fit, point_stats, SIDX
import exp_lib as L
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3

SEEDS = [0, 1, 2]
NPER, NTOT_PER_UNIT = 200, 3000
L1, L2 = 12.0, 0.25
RES = os.path.join(HERE, 'v5b_exp2_test_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def refit_and_test_stats(sd, cache):
    """re-run the exp2 fit (identical protocol/rng), save the model, and
    write test stats.  Sanity: dev train_idx must match the cached one."""
    out_path = os.path.join(HERE, f'v4_exp2_test_stats_s{sd}.npz')
    if os.path.exists(out_path):
        log(f'seed{sd}: test stats cached')
        return
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    ons = E2.detected_onsets(sd)
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
    old = np.load(os.path.join(HERE, f'v4_exp2_stats_s{sd}.npz'))['train_idx']
    assert np.array_equal(np.sort(tr), np.sort(old)), \
        f'seed{sd}: train_idx mismatch with the recorded exp2 run'
    log(f'seed{sd}: train_idx reproduces the recorded run ({len(tr)} rows)')
    t0 = time.time()
    gp, nret = gp_fit(W[tr], X[tr])
    log(f'seed{sd}: fitted ({time.time()-t0:.0f}s, retries={nret})')
    try:
        torch.save(gp, os.path.join(HERE, f'v4_exp2_gp_s{sd}.pt'))
        log(f'seed{sd}: model saved')
    except Exception as e:                      # save is best-effort
        log(f'seed{sd}: model save FAILED ({e}) — continuing')
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    unit_t = At[:, 0].astype(int); cyc_t = At[:, 1].astype(int)
    out = {}
    for u in np.unique(unit_t):
        rows = np.where(unit_t == u)[0]
        cyc_u = cyc_t[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        ucyc = np.unique(cyc_u)
        idx, cc = [], []
        for c in ucyc:
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx.append(r); cc.append(np.full(len(r), c))
        idx = np.concatenate(idx); cc = np.concatenate(cc)
        pred, dcv, ll = point_stats(gp, Wt[idx], Xt[idx])
        out[f'u{u}_cc'] = cc.astype(np.int32)
        out[f'u{u}_resid'] = (Xt[idx] - pred).astype(np.float32)
        out[f'u{u}_dc'] = dcv.astype(np.float32)
    np.savez(out_path, **out)
    log(f'seed{sd}: test stats written ({time.time()-t0:.0f}s total)')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5B EXP2 TEST — healthy-range ablation on test '
        '(11th-opening addendum, pre-registered scope)')
    cache = L.load_cache()
    for sd in SEEDS:
        refit_and_test_stats(sd, cache)

    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    dev_rm, t_rm, t_ns, Ps, Ts, Fs = [], [], [], [], [], []
    Ht = {sd: np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
          for sd in SEEDS}
    for sd in SEEDS:
        units, Zn, hrs = E2.tables(sd, -1e9)            # gate off, dev
        raw_mu_sg = None
        # recover mu/sg exactly as E2.tables (pooled over raw tables)
        Z = np.load(os.path.join(HERE, f'v4_exp2_stats_s{sd}.npz'))
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        raw = {}
        for u in units:
            cc = Z[f'u{u}_cc']; rs = Z[f'u{u}_resid']
            ur = H[f'dev_{u}_ucyc']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
            T = np.empty((len(ucyc), rs.shape[1]))
            for i, c in enumerate(ucyc):
                T[i] = trim25(rs[cc == c])
            raw[u] = T
        allr = np.concatenate([raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        dev_rm.append(float(np.sqrt(((P-T)**2).mean())))
        Zt = np.load(os.path.join(HERE, f'v4_exp2_test_stats_s{sd}.npz'))
        t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                          if k.endswith('_cc')})
        t_raw = {}
        for u in t_units:
            cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']
            ucyc = np.unique(cc)
            T2m = np.empty((len(ucyc), rs.shape[1]))
            for i, c in enumerate(ucyc):
                T2m[i] = trim25(rs[cc == c])
            t_raw[u] = T2m
        Zn_t = {u: (t_raw[u]-mu)/sg for u in t_units}
        HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P2, T2, F2, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P2-T2)**2).mean())))
        t_ns.append(nasa(P2, T2))
        Ps.append(P2); Ts.append(T2); Fs.append(F2)
        log(f'seed{sd}: dev LOO={dev_rm[-1]:.2f} beta={b} '
            f'test={t_rm[-1]:.2f} NASA={t_ns[-1]:.1f}')
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log(f'\nSANITY: dev LOO = {np.mean(dev_rm):.2f} ± {np.std(dev_rm):.2f} '
        f'(expected 7.61 ± 0.02)')
    log(f'healthy-range TEST RUL = {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  '
        f'NASA {np.mean(t_ns):.1f} ± {np.std(t_ns):.1f}')
    log('test per-trunc = ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    np.savez(os.path.join(HERE, 'v5b_exp2_test.npz'), P=P, T=T, F=F,
             t_rm=np.array(t_rm), t_ns=np.array(t_ns))
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
