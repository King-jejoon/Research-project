"""
exp_gate_ablation_nc.py — STAGE 2 of the new-chain rebuild: is the detcov
inspection gate (keep points with detcov >= V*) still useful once the raw-data
kNN cleaning is in place?

Two levels, both dev-only, both paired:

A. DETECTION — per-point ll/dc from newclean_stats_s{seed}.npz (3 seeds, model
   trained on the kNN-cleaned pool).  Cycle summary = LL quantile q in
   {70,75,80,85,90}, gate on (dc >= V*, V* = 15th pct of pooled dev detcov,
   absolute) vs off.  Frozen detector: 30 flight-hour baseline -> clip
   mu0-10*sigma0 -> mean-drop.  27 cases; Wilcoxon on |error|.

B. HI -> RUL — per-flight trim25 summaries from newclean_rul_s{seed}.npz,
   gated (trim) vs ungated (trim_ng).  Frozen HI config (HI_CFG,
   end_target=1.03), cycle-axis Stage-5 with beta re-selected by nested dev-LOO
   NASA per variant (the rule is frozen, the value is data-dependent).
   Metric: dev LOO truncation RUL RMSE / NASA over 9 units x 4 fracs x 3 seeds.

Decision rule (fixed in advance): keep the gate only if it beats gate-off
beyond seed noise on at least one level without hurting the other; otherwise
drop it (simpler chain).
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import (HI_CFG, FRACS, BETA_C, nasa, loo, train_model_tail)
from exp_resid_clean import meandrop

SEEDS = [0, 1, 2]
QS = [70, 75, 80, 85, 90]
RES = os.path.join(HERE, 'gate_ablation_nc_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


# ---------------- A. detection level ----------------
def load_det(sd):
    P = np.load(os.path.join(HERE, f'newclean_stats_s{sd}.npz'))
    R = np.load(os.path.join(HERE, f'newclean_rul_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in P.files if k.endswith('_cc')})
    out = {}
    for u in units:
        cc = P[f'u{u}_cc']; ll = P[f'u{u}_ll']; dc = P[f'u{u}_dc']
        ucyc_r = R[f'dev_{u}_ucyc']; dur = R[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ucyc_r)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        out[u] = dict(cc=cc, ll=ll, dc=dc, ucyc=ucyc,
                      dur=np.array([dur[pos[int(c)]] for c in ucyc], float),
                      onset=int(P[f'u{u}_onset'][0]))
    return out


def detect(curve, dur, ucyc):
    w = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
    mu0, sd0 = curve[:w].mean(), curve[:w].std(ddof=1) + 1e-8
    x = np.maximum(curve, mu0 - 10 * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def det_errors(D, Vs, q, gate):
    e = []
    for sd in SEEDS:
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                v = d['ll'][b]
                m = (d['dc'][b] >= Vs[sd]) if gate else np.ones(len(v), bool)
                if not m.any():
                    m = np.ones(len(v), bool)
                cur[i] = np.percentile(v[m], q)
            e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
    return np.array(e, float)


# ---------------- B. HI/RUL level ----------------
def load_hi(sd, variant):
    z = np.load(os.path.join(HERE, f'newclean_rul_s{sd}.npz'))
    units = sorted({int(k.split('_')[1]) for k in z.files
                    if k.startswith('dev_') and k.endswith('_ucyc')})
    raw = {u: z[f'dev_{u}_{variant}'] for u in units}
    hrs = {u: np.cumsum(z[f'dev_{u}_hours']) for u in units}
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    Z = {u: (raw[u] - mu) / sg for u in units}
    return units, Z, hrs


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('STAGE 2 — detcov gate ablation on the NEW chain (kNN-cleaned model)')

    # ---- A ----
    D = {sd: load_det(sd) for sd in SEEDS}
    Vs = {sd: float(np.percentile(np.concatenate([D[sd][u]['dc']
                                                  for u in D[sd]]), 15))
          for sd in SEEDS}
    log(f'  V* (15th pct pooled dev detcov): {[round(Vs[s],3) for s in SEEDS]}')
    log('')
    log('A. DETECTION (onset RMSE, cycles; 9 units x 3 seeds)')
    log(f'  {"gate":>5} | ' + ' | '.join(f'q{q:<4}' for q in QS))
    E = {}
    for gate in [True, False]:
        row = []
        for q in QS:
            E[(gate, q)] = det_errors(D, Vs, q, gate)
            row.append(np.sqrt((E[(gate, q)] ** 2).mean()))
        log(f'  {str(gate):>5} | ' + ' | '.join(f'{v:5.2f}' for v in row))
    a, b = E[(True, 75)], E[(False, 75)]
    p = st.wilcoxon(np.abs(a), np.abs(b), zero_method='zsplit').pvalue
    log(f'  paired at q75: gate {np.sqrt((a**2).mean()):.2f} vs no-gate '
        f'{np.sqrt((b**2).mean()):.2f}  (Wilcoxon p={p:.3f}, '
        f'gate better {int((np.abs(a)<np.abs(b)).sum())} worse '
        f'{int((np.abs(a)>np.abs(b)).sum())} of 27)')

    # ---- B ----
    log('')
    log('B. HI -> dev LOO truncation RUL (trim25 inputs, HI_CFG frozen, '
        'beta by nested dev-LOO NASA)')
    cfg = dict(HI_CFG); cfg['end_target'] = 1.03
    perseed = {}
    for variant, tag in [('trim', 'gated'), ('trim_ng', 'ungated')]:
        rmses, nasas, betas = [], [], []
        errpool = []
        for sd in SEEDS:
            units, Z, hrs = load_hi(sd, variant)
            np.random.seed(sd)
            him, _ = train_model_tail([Z[u] for u in units], **cfg)
            HIs = {u: him.forward(Z[u]).flatten() for u in units}
            P, T, F, bsel, _ = loo('cycle', units, HIs, hrs, BETA_C)
            rm = float(np.sqrt(((P - T) ** 2).mean()))
            rmses.append(rm); nasas.append(nasa(P, T)); betas.append(bsel)
            errpool.append(np.abs(P - T))
            log(f'  seed{sd} {tag:>7}: LOO RMSE={rm:.2f}  NASA={nasa(P,T):.0f}  '
                f'beta={bsel}')
        perseed[tag] = (np.array(rmses), np.array(nasas), betas,
                        np.concatenate(errpool))
    rg, ng = perseed['gated'], perseed['ungated']
    log(f'  gated  : RMSE {rg[0].mean():.2f} ± {rg[0].std():.2f}   '
        f'NASA {rg[1].mean():.0f}   beta {rg[2]}')
    log(f'  ungated: RMSE {ng[0].mean():.2f} ± {ng[0].std():.2f}   '
        f'NASA {ng[1].mean():.0f}   beta {ng[2]}')
    pw = st.wilcoxon(rg[3], ng[3], zero_method='zsplit').pvalue
    log(f'  paired |error| over {len(rg[3])} cases: Wilcoxon p={pw:.3f}  '
        f'(gated better {int((rg[3]<ng[3]).sum())}, worse '
        f'{int((rg[3]>ng[3]).sum())})')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
