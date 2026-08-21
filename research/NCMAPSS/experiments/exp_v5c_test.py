"""
exp_v5c_test.py — PRE-REGISTERED one-time test run = the THIRTEENTH DS03
test opening, disclosed.  Trigger: proposed healthy-range chain frozen on
dev (exp_v5c_dev; user decisions 2026-08-14: window = current conditional-
gate onsets, downstream UNGATED).

Pre-registration, frozen BEFORE this run:
  (a) proposed chain -> test truncation RUL, NASA, per-trunc; paired |err|
      vs the official re-frozen chain (v5b_test_batch chain_*) and vs the
      old healthy-range ablation (v5b_exp2_test);
  (b) secondary — UNGATED onset re-detection on test with the v5c GP:
      RMSE vs the official conditional 8.54 (official per-case errors
      recomputed from cached v4_test_stats as pairing sanity), paired
      Wilcoxon over the 18 cases; dev onsets are circular (the window
      came from the 1-pass onsets) and are reported as reference only;
  (c) every number reported regardless of direction; unit n=6 caveat.
Sanity gate: the dev side must reproduce v5c_dev (mean rms within 0.02)
BEFORE any test file is read.  GPs loaded from v5c_gp_s{sd}.pt — no
refits (fallback on load failure: refit from the recorded train_idx).
Outputs: v5c_test_results.txt, v5c_test_stats_s{sd}.npz, v5c_test.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
from exp_v5c_dev import tables, NPER, L1, L2, VD, VS, NB
from exp_v4_c3_detcov import fit as gp_fit, point_stats, SIDX
from exp_v4_setcmp import detect
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3

SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'v5c_test_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load_gp(sd, cache):
    p = os.path.join(HERE, f'v5c_gp_s{sd}.pt')
    try:
        try:
            gp = torch.load(p, weights_only=False)
        except TypeError:                        # older torch: no kwarg
            gp = torch.load(p)
        log(f'seed{sd}: gp loaded')
        return gp
    except Exception as e:
        log(f'seed{sd}: gp load FAILED ({e}) — refitting from train_idx')
        tr = np.load(os.path.join(HERE, f'v5c_stats_s{sd}.npz'))['train_idx']
        W, X = cache['W_dev'], cache['X_s_dev'][:, SIDX]
        t0 = time.time()
        gp, nret = gp_fit(W[tr], X[tr])
        log(f'seed{sd}: refitted ({time.time()-t0:.0f}s, retries={nret})')
        return gp


def build_test_stats(sd, cache):
    out_path = os.path.join(HERE, f'v5c_test_stats_s{sd}.npz')
    if os.path.exists(out_path):
        log(f'seed{sd}: test stats cached')
        return
    gp = load_gp(sd, cache)
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    unit_t = At[:, 0].astype(int); cyc_t = At[:, 1].astype(int)
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    t0 = time.time()
    out = {}
    for u in np.unique(unit_t):
        rows = np.where(unit_t == u)[0]
        cyc_u = cyc_t[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)   # chain convention
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
        out[f'u{u}_ll'] = ll.astype(np.float32)
        for k in ('ucyc', 'hours', 'onset'):     # data properties, copied
            out[f'u{u}_{k}'] = Zt[f'u{u}_{k}']
    np.savez(out_path, **out)
    log(f'seed{sd}: test stats written ({time.time()-t0:.0f}s)')


def test_tables(sd):
    """UNGATED test tables from the v5c stats."""
    Zt = np.load(os.path.join(HERE, f'v5c_test_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                    if k.endswith('_cc')})
    raw = {}
    for u in units:
        cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']
        ucyc = Zt[f'u{u}_ucyc']
        keep = np.isin(ucyc, np.unique(cc))
        ucyc = ucyc[keep]
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[cc == c])
        raw[u] = T
    return units, raw


def onset_errs(path_fmt, sd, gated):
    """per-case onset errors from a test-stats file (frozen detector)."""
    Zt = np.load(os.path.join(HERE, path_fmt.format(sd=sd)))
    units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                    if k.endswith('_cc')})
    e = []
    for u in units:
        cc = Zt[f'u{u}_cc']; ll = Zt[f'u{u}_ll']
        ucyc = Zt[f'u{u}_ucyc']; hours = Zt[f'u{u}_hours'].astype(float)
        keep = np.isin(ucyc, np.unique(cc))
        ucyc = ucyc[keep]; dur = hours[keep]
        V = None
        if gated:
            dc = Zt[f'u{u}_dc']
            nb = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
            V = VS if nb < NB else VD
        cur = np.empty(len(ucyc))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b if V is None else (b & (dc >= V))
            if not m.any():
                m = b
            cur[i] = np.percentile(ll[m], 75)
        e.append(detect(cur, dur, ucyc) - int(Zt[f'u{u}_onset'][0]))
    return np.array(e, float)


def dev_onset_errs(sd):
    """proposed-GP dev onsets, ungated (circular — reference only)."""
    Z = np.load(os.path.join(HERE, f'v5c_stats_s{sd}.npz'))
    Zo = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files
                    if k.endswith('_cc')})
    e = []
    for u in units:
        cc = Z[f'u{u}_cc']; ll = Z[f'u{u}_ll']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        dur = np.array([durs[pos[int(c)]] for c in ucyc], float)
        cur = np.array([np.percentile(ll[cc == c], 75) for c in ucyc])
        e.append(detect(cur, dur, ucyc) - int(Zo[f'u{u}_onset'][0]))
    return np.array(e, float)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5C TEST — THIRTEENTH DS03 test opening, pre-registered: proposed '
        'healthy-range chain (ungated downstream), frozen on dev')
    exp = np.load(os.path.join(HERE, 'v5c_dev.npz'))
    exp_rm = float(np.mean(exp['rms']))

    # dev side first — sanity gate before any test file is touched
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    dev_rm, side = [], []
    for sd in SEEDS:
        units, raw, Zn, hrs, mu, sg = tables(sd)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        dev_rm.append(float(np.sqrt(((P - T) ** 2).mean())))
        side.append((him, mu, sg, HIs, hrs, b, units))
        log(f'seed{sd}: dev LOO={dev_rm[-1]:.2f} '
            f'shape={"OK" if ok else "FAIL"} beta={b}')
    log(f'SANITY dev = {np.mean(dev_rm):.2f} (frozen v5c_dev {exp_rm:.2f})')
    assert abs(np.mean(dev_rm) - exp_rm) <= 0.02, 'dev sanity FAILED — stop'

    # test contact starts here
    cache = L.load_cache()
    for sd in SEEDS:
        build_test_stats(sd, cache)

    t_rm, t_ns, Ps, Ts, Fs = [], [], [], [], []
    for sd in SEEDS:
        him, mu, sg, HIs, hrs, b, units = side[sd]
        t_units, t_raw = test_tables(sd)
        Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
        HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P2, T2, F2, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P2 - T2) ** 2).mean())))
        t_ns.append(nasa(P2, T2))
        Ps.append(P2); Ts.append(T2); Fs.append(F2)
        log(f'seed{sd}: test={t_rm[-1]:.2f} NASA={t_ns[-1]:.1f}')
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log('')
    log(f'V5C test RUL = {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  '
        f'NASA {np.mean(t_ns):.1f} ± {np.std(t_ns):.1f}')
    log('test per-trunc = ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    Zb = np.load(os.path.join(HERE, 'v5b_test_batch.npz'))
    assert np.allclose(T, Zb['chain_T']), 'pairing misaligned vs chain'
    p1 = st.wilcoxon(np.abs(P - T), np.abs(Zb['chain_P'] - Zb['chain_T']),
                     zero_method='zsplit').pvalue
    Z2 = np.load(os.path.join(HERE, 'v5b_exp2_test.npz'))
    assert np.allclose(T, Z2['T']), 'pairing misaligned vs exp2'
    p2 = st.wilcoxon(np.abs(P - T), np.abs(Z2['P'] - Z2['T']),
                     zero_method='zsplit').pvalue
    log(f'paired |err| (72 preds, unit n=6 caveat): vs official chain 7.52 '
        f'p={p1:.3f} | vs old healthy ablation 7.26 p={p2:.3f}')

    # (b) secondary — onset re-detection on test
    e_off = np.concatenate([onset_errs('v4_test_stats_s{sd}.npz', sd, True)
                            for sd in SEEDS])
    r_off = float(np.sqrt((e_off ** 2).mean()))
    log('')
    log(f'onset pairing sanity: official conditional test RMSE = '
        f'{r_off:.2f} (expected 8.54)')
    assert abs(r_off - 8.54) <= 0.02, 'onset pairing sanity FAILED'
    e_prop = np.concatenate([onset_errs('v5c_test_stats_s{sd}.npz', sd,
                                        False) for sd in SEEDS])
    r_prop = float(np.sqrt((e_prop ** 2).mean()))
    try:
        po = st.wilcoxon(np.abs(e_prop), np.abs(e_off),
                         zero_method='zsplit').pvalue
        po = f'{po:.3f}'
    except ValueError:                           # identical error vectors
        po = '1.000 (identical)'
    log(f'onset secondary: proposed ungated test RMSE = {r_prop:.2f} '
        f'vs official 8.54, paired p={po} (18 cases)')
    e_dev = np.concatenate([dev_onset_errs(sd) for sd in SEEDS])
    log(f'onset dev reference (circular, 2-pass): proposed ungated dev '
        f'RMSE = {float(np.sqrt((e_dev ** 2).mean())):.2f} vs official 5.21')

    np.savez(os.path.join(HERE, 'v5c_test.npz'), P=P, T=T, F=F,
             t_rm=np.array(t_rm), t_ns=np.array(t_ns),
             dev_rm=np.array(dev_rm), e_on_prop=e_prop, e_on_off=e_off,
             e_on_dev=e_dev)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
