"""exp_v5c_raw_resummary.py — RE-SUMMARY of the eleventh-opening raw-input
ablation (raw per-cycle sensor means -> frozen l12 HI -> beta -> test RUL),
identical computation, only the per-prediction arrays are archived this
time so per-level NASA scores can be tabulated.  Sanity: must reproduce the
recorded 8.31 +/- 0.35 / NASA 22.6.  Output: v5c_raw_resummary.npz"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, train_model_tail
from exp_v4_final import med3

SEEDS = [0, 1, 2]; NPER = 200; L1, L2 = 12.0, 0.25
SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
cache = L.load_cache()
tabs = {s: (cache[f'X_s_{s}'][:, SIDX], cache[f'A_{s}'][:, 0].astype(int),
            cache[f'A_{s}'][:, 1].astype(int)) for s in ['dev', 'test']}


def dev_raw(sd):
    X, unit, cyc = tabs['dev']
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    raw, hrs = {}, {}
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]; cyc_u = cyc[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cyc_u) if int(c) in pos])
        T = np.empty((len(ucyc), X.shape[1]))
        for i, c in enumerate(ucyc):
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            T[i] = X[r].mean(0)
        raw[int(u)] = T
        hrs[int(u)] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
    units = sorted(raw); allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, {u: (raw[u] - mu) / sg for u in units}, hrs, mu, sg


def test_raw(sd):
    X, unit, cyc = tabs['test']
    raw = {}
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]; cyc_u = cyc[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        ucyc = np.unique(cyc_u)
        T = np.empty((len(ucyc), X.shape[1]))
        for i, c in enumerate(ucyc):
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            T[i] = X[r].mean(0)
        raw[int(u)] = T
    return sorted(raw), raw


cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
t_rm, t_ns, Ps, Ts, Fs = [], [], [], [], []
for sd in SEEDS:
    units, Zn, hrs, mu, sg = dev_raw(sd)
    np.random.seed(sd)
    him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
    P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
    tu, t_r = test_raw(sd)
    Zn_t = {u: (t_r[u] - mu) / sg for u in tu}
    HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in tu}
    cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
    ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in tu}
    P2, T2, F2, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
    t_rm.append(float(np.sqrt(((P2 - T2) ** 2).mean()))); t_ns.append(nasa(P2, T2))
    Ps.append(P2); Ts.append(T2); Fs.append(F2)
    print(f'seed{sd}: test={t_rm[-1]:.2f} NASA={t_ns[-1]:.1f} beta={b}', flush=True)
print(f'raw re-summary: {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  NASA {np.mean(t_ns):.1f} '
      f'(recorded 8.31 ± 0.35 / 22.6)')
assert abs(np.mean(t_rm) - 8.31) <= 0.02 and abs(np.mean(t_ns) - 22.6) <= 0.2, 'does not reproduce'
np.savez(os.path.join(HERE, 'v5c_raw_resummary.npz'), P=np.concatenate(Ps), T=np.concatenate(Ts),
         F=np.concatenate(Fs), t_rm=np.array(t_rm), t_ns=np.array(t_ns))
print('saved v5c_raw_resummary.npz')
