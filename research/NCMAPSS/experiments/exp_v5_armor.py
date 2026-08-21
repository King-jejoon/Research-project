"""
exp_v5_armor.py — downstream armor-removal grid, MOGP row re-based on the
coverage-conditional gate (user decision 2026-08-10).  Competitor rows and
the ungated MOGP row involve no gate, so they are unchanged from
exp_v4_frag_summary and are NOT recomputed here.

Cells: summary {trim25, mean} x HI post-filter {med3, none}, dev LOO
truncation RUL, 3 seeds.  trim25+med3 must reproduce the final chain
7.10 +/- 0.07 — sanity gate.  Dev only, cached stats, no test contact.
Outputs: v5_armor_results.txt, v5_armor.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_hi import trim25, SEEDS
from exp_v4_final import med3

VD, VS, NB = 20.2, 19.0, 10
L1, L2 = 8.0, 0.25
RES = os.path.join(HERE, 'v5_armor_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def nb_of(dur):
    return max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))


def summarize(rows, how):
    return trim25(rows) if how == 'trim25' else rows.mean(0)


def cond_tables(sd, how):
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        dur = np.array([durs[pos[int(c)]] for c in ucyc], float)
        V = VS if nb_of(dur) < NB else VD
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= V)
            T[i] = summarize(rs[m if m.any() else b], how)
        raw[u] = T
        hrs[u] = np.cumsum(dur)
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, {u: (raw[u] - mu) / sg for u in units}, hrs


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5 ARMOR — MOGP conditional-gate row of the armor-removal grid')
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    out = {}
    for how in ['trim25', 'mean']:
        for filt in ['med3', 'none']:
            rmses, oks = [], []
            for sd in SEEDS:
                units, Zn, hrs = cond_tables(sd, how)
                np.random.seed(sd)
                him, _ = train_model_tail([Zn[u] for u in units], **cfg)
                HIs = {u: him.forward(Zn[u]).flatten() for u in units}
                if filt == 'med3':
                    HIs = {u: med3(h) for u, h in HIs.items()}
                ok, _ = shape_ok(HIs)
                P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
                rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
                oks.append(ok)
            out[(how, filt)] = (float(np.mean(rmses)), float(np.std(rmses)),
                                int(np.sum(oks)))
            log(f'  {how:>6}+{filt:<5}: RUL {np.mean(rmses):.2f} ± '
                f'{np.std(rmses):.2f}  shape {int(np.sum(oks))}/3')
    log('')
    ref = out[('trim25', 'med3')]
    log(f'SANITY trim25+med3 = {ref[0]:.2f} ± {ref[1]:.2f} '
        f'(expected 7.10 ± 0.07)')
    np.savez(os.path.join(HERE, 'v5_armor.npz'),
             **{f'{h}_{f}': np.array(v) for (h, f), v in out.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
